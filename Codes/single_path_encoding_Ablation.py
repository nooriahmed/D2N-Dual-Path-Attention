
# Single-path U-Net model with Domain Adaptation
def single_path_unet_with_da(input_shape=(60, 60, 3), dropout_rate=0.5, lambda_=1.0):
    inputs = Input(input_shape)

    # Encoder (single path)
    e1 = SeparableConv2D(64, (3, 3), padding='same')(inputs)
    e1 = BatchNormalization()(e1)
    e1 = PReLU()(e1)
    e1 = MaxPooling2D((2, 2))(e1)

    e2 = SeparableConv2D(128, (3, 3), padding='same')(e1)
    e2 = BatchNormalization()(e2)
    e2 = PReLU()(e2)
    e2 = MaxPooling2D((2, 2))(e2)

    # Bottleneck with residual
    shortcut = e2
    x = SeparableConv2D(256, (3, 3), padding='same')(e2)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Dropout(dropout_rate)(x)
    x = SeparableConv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = PReLU()(x)

    # Decoder
    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, e1])
    x = SeparableConv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, inputs])
    x = SeparableConv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    # Segmentation output
    segmentation_outputs = Conv2D(1, (1, 1), activation='sigmoid', name='segmentation_output')(x)

    # Domain adaptation path
    domain_grl = GradientReversalLayer(lambda_)(x)
    domain_output = domain_classifier(domain_grl)
    domain_output = Dense(1, activation='sigmoid', name='domain_output')(domain_output)

    model = Model(inputs=inputs, outputs=[segmentation_outputs, domain_output])
    return model
