import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2

class LinearDense(tf.keras.layers.Layer):
    def __init__(self, source_layer, **kwargs):
        super().__init__(**kwargs)
        # Reference the existing kernel and bias directly
        # No weight copying — we point to the same variables
        self.w = source_layer.kernel
        self.b = source_layer.bias

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class HeatMap:
    def __init__(self, model, img_path, class_names, target_class=None,preprocess=None):
        # loads model with compile=false so we avoid custom loss / optimizer conflicts
        # since we only need it for inference (Grad-CAM), not training
        # if model is already loaded                                 # load model from a path
        self.model = model if isinstance(model, tf.keras.Model) else load_model(model, compile=False)
        #for custom scaling
        #if none the model will try to detect it from the pre-trained model,
        #if no pre-trained model,it will try to detect for internal scaling
        #if non it will /255(standerd)
        self.user_preprocess = preprocess
        # automatically detect if the loaded model contains a pre-trained backbone
        # (like vgg16, resnet50,...etc)
        # check detect_backbone_submodel() function for detection logic
        self.pre_trained_model = self.detect_backbone_submodel()
        # if no backbone was detected
        # it means the model is either custom(no pre-trained model) or not nested
        # so we use the full model directly
        if self.pre_trained_model is None:
            self.pre_trained_model = self.model
            # no backbone name available
            # so we set it to None (no special preprocessing needed)
            self.backbone_name = None
        # if a backbone exists
        # we extract its name (resnet50, vgg16, etc.)
        # this helps us choose the correct preprocessing function later
        else:
            self.backbone_name = self.pre_trained_model.name.lower()
        # store image path
        self.img_path = img_path
        # some models may have multiple inputs
        # example: [(None,224,224,3), (None,10)]
        # so we check if input_shape is a list
        if isinstance(self.model.input_shape, list):
            # take the first input (image input)
            input_shape = self.model.input_shape[0]
        else:
            # normal single-input model
            input_shape = self.model.input_shape
        # extract image size from model input shape
        # tensorflow input shape format: (batch, height, width, channels)
        # so by calling [1:3] we get (height, width)
        # which is the image_size the model was trained on
        if None in input_shape[1:3]:
            # if height or width is None
            # it means the model supports dynamic input size
            # so we cannot auto-detect image size safely
            raise ValueError("Dynamic input size detected. Please specify image_size manually.")
        self.image_size = input_shape[1:3]
        print(f"Detected Image Size = {self.image_size}")
        print(f"Detected Backbone = {self.backbone_name}")
        # store class names for prediction interpretation
        self.class_names = class_names
        # optional target class
        # if None → we use predicted class
        # if provided → Grad-CAM will focus on that specific class
        self.target_class = target_class



    def overlay_heatmap(self, alpha=0.4):
        try:
            # compute Grad-CAM heatmap
            # check compute_gradcam() function for detailed logic
            heatmap = self.compute_gradcam()

            # validate heatmap output
            # ensure it is not None, is a numpy array, and not empty
            if heatmap is None or not isinstance(heatmap, np.ndarray) or heatmap.size == 0:
                raise ValueError("Grad-CAM heatmap is empty (compute_gradcam returned None/empty).")

            # read original image using OpenCV in BGR format
            img = cv2.imread(self.img_path, cv2.IMREAD_COLOR)

            # ensure image was loaded correctly
            # if path is wrong or image is corrupted, cv2.imread returns None
            if img is None or img.size == 0:
                raise ValueError(f"cv2.imread failed. Bad path or unreadable image: {self.img_path}")

            # ensure heatmap is converted to numpy array
            heatmap = np.asarray(heatmap)

            # Grad-CAM heatmap must be 2D (height, width)
            # if it has more dimensions, something went wrong in compute_gradcam()
            if heatmap.ndim != 2:
                raise ValueError(f"Heatmap must be 2D, got shape: {heatmap.shape}")

            # convert heatmap to float32
            # this ensures stable normalization later
            heatmap = heatmap.astype(np.float32)

            # get original image height and width
            h, w = img.shape[:2]

            # resize heatmap to match original image size
            # because Grad-CAM output matches model input size
            # not necessarily the original image size
            heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

            # normalize heatmap values to range [0, 255]
            # this is required for applying color map
            heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)

            # convert heatmap to uint8
            # cv2.applyColorMap requires 8-bit image
            heatmap = heatmap.astype(np.uint8)

            # apply JET color map
            # low importance --> blue
            # high importance --> red
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # blend original image and colored heatmap
            # alpha controls visibility of original image
            # (img * alpha) + (heatmap * (1 - alpha))
            # both images are BGR, so blending is correct
            superimposed = cv2.addWeighted(img, alpha, heatmap_color, 1 - alpha, 0)

            # final safety check
            # ensure overlay result is valid
            if superimposed is None or superimposed.size == 0:
                raise ValueError("Overlay result is empty.")

            # return final Grad-CAM visualization
            return superimposed

        except Exception as E:
            print(E)

    def build_grad_model(self):
        conv_layer_name = self.last_conv_layer_get(self.pre_trained_model)
        last_layer = self.model.layers[-1]

        # Check if final layer has a non-linear activation (sigmoid, softmax, etc.)
        has_activation = (
                hasattr(last_layer, 'activation') and
                last_layer.activation != tf.keras.activations.linear
        )

        # --- CASE 1: Nested backbone (VGG16, ResNet, etc.) ---
        if self.pre_trained_model is not self.model:

            backbone_grad_model = tf.keras.models.Model(
                inputs=self.pre_trained_model.inputs,
                outputs=[
                    self.pre_trained_model.get_layer(conv_layer_name).output,
                    self.pre_trained_model.output
                ]
            )

            outer_input = self.model.inputs
            conv_out, backbone_out = backbone_grad_model(outer_input)

            x = backbone_out
            head_started = False
            for layer in self.model.layers:
                if layer.name == self.backbone_name:
                    head_started = True
                    continue
                if head_started:
                    if layer == last_layer and has_activation:
                        # Use tracked LinearDense instead of raw tf.matmul
                        x = LinearDense(last_layer)(x)
                    else:
                        x = layer(x)

            grad_model = tf.keras.models.Model(
                inputs=outer_input,
                outputs=[conv_out, x]
            )

        # --- CASE 2: Flat custom model ---
        else:
            if has_activation:
                pre_act = self.model.layers[-2].output
                # Use tracked LinearDense instead of raw tf.matmul
                logit_output = LinearDense(last_layer)(pre_act)
            else:
                logit_output = self.model.output

            grad_model = tf.keras.models.Model(
                inputs=self.model.inputs,
                outputs=[
                    self.model.get_layer(conv_layer_name).output,
                    logit_output
                ]
            )

        return grad_model

    def compute_gradcam(self):
        try:
            preprocessed_image = self.preprocess_image()
            idx, name, conf, probs = self.predict(preprocessed_image)
            grad_model = self.build_grad_model()

            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(preprocessed_image)

                if predictions.shape[-1] == 1:
                    explain_class = int(self.target_class) if self.target_class is not None else int(idx)
                    raw = predictions[:, 0]
                    loss = tf.math.log(raw + 1e-10) if explain_class == 1 else tf.math.log(1.0 - raw + 1e-10)
                else:
                    class_index = int(self.target_class) if self.target_class is not None else tf.argmax(predictions[0])
                    loss = tf.math.log(predictions[:, class_index] + 1e-10)

            grads = tape.gradient(loss, conv_outputs)

            # Guard against None gradients before any operations
            # This happens when the conv layer is disconnected from the loss in the graph
            if grads is None:
                raise ValueError(
                    f"Gradients are None for layer targeted by Grad-CAM. "
                    f"The conv layer output may be disconnected from the loss. "
                    f"Check build_grad_model() graph connectivity."
                )

            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs.numpy()[0]
            pooled_grads = pooled_grads.numpy()

            for i in range(pooled_grads.shape[-1]):
                conv_outputs[:, :, i] *= pooled_grads[i]

            heatmap = np.mean(conv_outputs, axis=-1)
            heatmap = np.maximum(heatmap, 0)
            heatmap /= (np.max(heatmap) + 1e-10)

            return heatmap

        except Exception as E:
            raise ValueError(f"Error in compute_gradcam: {E}")
    def get_preprocess_function(self):
        mapping = {
            "vgg16": tf.keras.applications.vgg16.preprocess_input,
            "vgg19": tf.keras.applications.vgg19.preprocess_input,
            "resnet": tf.keras.applications.resnet.preprocess_input,
            "resnet50": tf.keras.applications.resnet.preprocess_input,
            "resnetv2": tf.keras.applications.resnet_v2.preprocess_input,
            "mobilenet": tf.keras.applications.mobilenet.preprocess_input,
            "mobilenetv2": tf.keras.applications.mobilenet_v2.preprocess_input,
            "mobilenetv3": tf.keras.applications.mobilenet_v3.preprocess_input,
            "efficientnet": tf.keras.applications.efficientnet.preprocess_input,
            "efficientnetv2": tf.keras.applications.efficientnet_v2.preprocess_input,
            "densenet": tf.keras.applications.densenet.preprocess_input,
            "inception": tf.keras.applications.inception_v3.preprocess_input,
            "inceptionresnet": tf.keras.applications.inception_resnet_v2.preprocess_input,
            "xception": tf.keras.applications.xception.preprocess_input,
            "nasnet": tf.keras.applications.nasnet.preprocess_input,
            "convnext": tf.keras.applications.convnext.preprocess_input,
            "regnet": tf.keras.applications.regnet.preprocess_input,
        }
        p = self.user_preprocess

        # 1) User provided a callable -> use it
        if callable(p):
            print("Using USER preprocessing function.")
            return p

        # 2) User provided a string -> look it up
        if isinstance(p, str):
            key = p.lower().strip()
            fn = mapping.get(key)
            if fn is None:
                raise ValueError(
                    f"Unknown preprocess='{p}'. Pass a callable or one of: {sorted(mapping.keys())}"
                )
            print(f"Using USER preprocess='{p}'.")
            return fn

        # 3) No user preprocess -> try backbone_name
        if self.backbone_name:
            key = self.backbone_name.lower().strip()

            # Direct match
            fn = mapping.get(key)
            if fn is not None:
                print(f"Using backbone preprocessing for '{self.backbone_name}'.")
                return fn

            # Fuzzy match (handles names like "resnet50_backbone", "vgg16_backbone")
            for k, v in mapping.items():
                if k in key:
                    print(f"Using backbone preprocessing matched '{k}' from '{self.backbone_name}'.")
                    return v

        # 4) Nothing found
        return None

    def preprocess_image(self):
        try:
            # check if the image path exists before trying to read it
            # this avoids cv2.imread returning None silently
            if not os.path.exists(self.img_path):
                print(f"Error : File Not Found {self.img_path}")
                raise FileNotFoundError

            # read image using OpenCV (default format is BGR)
            img = cv2.imread(self.img_path, cv2.IMREAD_COLOR)

            # convert image from BGR to RGB
            # because most TensorFlow / Keras models expect RGB input
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # resize image to the expected model input size
            img = cv2.resize(img, self.image_size)
            # so output shape becomes (1, H, W, 3)
            img = np.expand_dims(img, axis=0)
            # if backbone_name exists:
            # it means we detected a known pre-trained backbone (resnet, vgg, ...etc)
            # so we should use the backbone preprocess_input
            if self.backbone_name or self.user_preprocess:

                # get preprocessing function based on backbone name
                # example: resnet50 --> tf.keras.applications.resnet.preprocess_input
                # check get_preprocess_function() for mapping logic
                process_func = self.get_preprocess_function()

                # if we found a matching preprocessing function
                # we apply it directly and return the processed image
                if process_func:
                    return process_func(img)

            # if no backbone preprocessing is used:
            # we check if the model contains an internal Rescaling layer
            # some models already divide by 255 internally, so we should not do it twice
            has_internal_rescaling = False

            for layer in self.model.layers:
                # detect Keras Rescaling layer
                # example: tf.keras.layers.Rescaling(1./255)
                if isinstance(layer, tf.keras.layers.Rescaling):
                    has_internal_rescaling = True
                    print(f"Internal scaling detected ({layer.scale}). Passing raw pixels.")
                    print("  offset:", layer.offset)
                    break
            # final scaling decision:
            # if the model has internal rescaling --> pass raw pixels (0..255)
            # if not --> manually scale to [0, 1] by dividing by 255
            if has_internal_rescaling:
                return img
            else:
                print("No internal scaling or backbone found. Manually scaling to [0, 1].")
                return img / 255.0

        except Exception as E:
            raise ValueError(f"{E}")

    def last_conv_layer_get(self, model):
        # iterate over model layers in reverse order
        # because we want the LAST convolution layer (closest to output)
        for layer in reversed(model.layers):
            # check if current layer is a Conv2D or DepthwiseConv2D
            # DepthwiseConv2D is used in models like MobileNet
            # Conv2D is used in models like VGG, ResNet, etc.
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                # print detected layer name (for debugging)
                print(f"Layer Name : {layer.name}")

                # return the name of the last convolution layer found
                return layer.name

            # some models (like ResNet, EfficientNet, etc.)
            # may contain nested models inside the main model
            # so if the layer itself is a Model, we inspect its internal layers
            if isinstance(layer, tf.keras.Model):

                # again iterate in reverse order to find the last conv layer inside the sub-model
                for sub_layer in reversed(layer.layers):

                    # check if sub-layer is Conv2D or DepthwiseConv2D
                    if isinstance(sub_layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                        # print detected last convolution layer inside nested model
                        print(f"Last Convo Layer : {sub_layer.name}")

                        # return the name of that convolution layer
                        return sub_layer.name

        # if no convolution layer was found
        # return None (Grad-CAM cannot be computed without a conv layer)
        print("No Convolution layer was found ")
        return None

    def predict(self, img):
        try:
            preds = self.model(img, training=False).numpy()[0]
            if preds.shape[-1] == 1:
                score = float(preds[0])
                idx = 1 if score >= 0.5 else 0
                name = self.class_names[idx] if self.class_names else str(idx)
                conf = score if idx == 1 else 1.0 - score
                probs = {(self.class_names[0] if self.class_names else "0"): round(1 - score, 4),
                         (self.class_names[1] if len(self.class_names) > 1 else "1"): round(score, 4)}
                print(f"Class: {idx} {name}  Confidence: {conf * 100:.2f}%")
            else:
                idx = int(np.argmax(preds))
                conf = float(preds[idx])
                probs = {(self.class_names[i] if i < len(self.class_names) else str(i)): round(float(p), 4)
                         for i, p in enumerate(preds)}
                name = self.class_names[idx] if self.class_names and idx < len(self.class_names) else str(idx)
                print(f"Class: {idx} ({name})")
                print(f"Confidence: {conf * 100:.2f}%")
            return idx, name, conf, probs

        except Exception as e:
            raise ValueError(f"Predict Error: {e}")

    def detect_backbone_submodel(self):

        # search for nested models inside the top-level model
        # many architectures (ResNet, MobileNet, EfficientNet, etc.)
        # are wrapped as a backbone model inside a larger classification model
        # so we look for layers that are themselves instances of tf.keras.Model
        candidates = [l for l in self.model.layers if isinstance(l, tf.keras.Model)]

        # if no nested models were found
        # it means:
        # - either the model is fully custom
        # - or the backbone is not wrapped as a separate submodel
        # in this case we treat the full model as the backbone
        if not candidates:
            print(f"No nested backbone. Using top model name: {self.model.name}")
            return None

        # if nested models exist:
        # we assume one of them is the backbone and the others (if any)
        # are small wrapper blocks or utility layers

        # heuristic strategy:
        # choose the nested model with the highest number of parameters
        # because:
        # - backbone usually contains most of the parameters
        # - classification head is typically small (Dense layers)
        # so the largest nested model is very likely the backbone
        backbone = max(candidates, key=lambda m: m.count_params())

        print(f"Detected Model : {backbone.name}")

        # return detected backbone model
        return backbone



    def save_heat_img(self, name, output_img):
        try:
            # saving the heatmap Image
            folder_name = 'heatmap'
            os.makedirs(folder_name, exist_ok=True)

            if output_img is None or (isinstance(output_img, np.ndarray) and output_img.size == 0):
                raise ValueError("save_heat_img got an empty output_img (None/empty).")

            save_path = os.path.join(folder_name, name)
            ok = cv2.imwrite(save_path, output_img)

            if not ok:
                raise IOError(f"cv2.imwrite failed for path: {save_path}")

            print(f"Successfully saved heatmap to: {save_path}")
        except Exception as E:
            raise ValueError(f"{E}")

