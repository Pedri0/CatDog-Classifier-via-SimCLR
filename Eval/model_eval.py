import tensorflow.compat.v2 as tf
from absl import flags

import resnet_eval as resnet

FLAGS = flags.FLAGS

########### UTILFUNCTIONS #########################################
def add_weight_decay(model):
    # Weight decay are taking care of by optimizer for these cases.
    # Except for supervised head, which will be added here.
    l2_losses = [tf.nn.l2_loss(v) for v in model.trainable_variables if 'head_supervised' in v.name and
        'bias' not in v.name]
    if l2_losses:
        return FLAGS.weight_decay * tf.add_n(l2_losses)
    else:
        return 0

###################################################################

class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, num_classes, use_bias=True, use_bn=False, name='linear_layer', **kwargs):
        # Note: use_bias is ignored for the dense layer when use_bn =True. However, it is still used for batch norm
        super(LinearLayer, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.use_bn = use_bn
        self._name = name
        if callable(self.num_classes):
            num_classes = -1
        else:
            num_classes = self.num_classes
        self.dense = tf.keras.layers.Dense(num_classes, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        use_bias=use_bias and not self.use_bn)
        if self.use_bn:
            self.bn_relu = resnet.BatchNormRelu(relu=False, center=use_bias)

    def build(self, input_shape):
        if callable(self.num_classes):
            self.dense.units = self.num_classes(input_shape)
        super(LinearLayer,self).build(input_shape)

    def call(self, inputs, training):
        assert inputs.shape.ndims == 2, inputs.shape
        inputs = self.dense(inputs)
        if self.use_bn:
            inputs = self.bn_relu(inputs, training=training)
        return inputs

class ProjectionHead(tf.keras.layers.Layer):
    #using nonlinear projectionHead
    def __init__(self, **kwargs):
        self.linear_layers = []
        for j in range(FLAGS.num_proj_layers):
            if j != FLAGS.num_proj_layers - 1:
                #for the middle layers, use bias and relu for the output
                self.linear_layers.append(LinearLayer(num_classes=lambda input_shape: int(input_shape[-1]),
                use_bias=True, use_bn=True, name='nl_%d' % j))
            else:
                #for the final layer, neither bias nor relu is used
                self.linear_layers.append(LinearLayer(num_classes=FLAGS.proj_out_dim, use_bias=False, use_bn=True, name='nl_%d' %j))
        
        super(ProjectionHead, self).__init__(**kwargs)
    
    def call(self, inputs, training):
        hiddens_list = [tf.identity(inputs, 'proj_head_input')]
        for j in range(FLAGS.num_proj_layers):
            hiddens = self.linear_layers[j](hiddens_list[-1], training)
            if j!= FLAGS.num_proj_layers - 1:
                #for the middle layers, use bias and relu for the output.
                hiddens = tf.nn.relu(hiddens)
            hiddens_list.append(hiddens)

        #The element is the input of the finetune head
        return hiddens_list[FLAGS.ft_proj_selector]

class SupervisedHead(tf.keras.layers.Layer):
    def __init__(self, num_classes, name='head_supervised', **kwargs):
        super(SupervisedHead, self).__init__(name=name, **kwargs)
        self.linear_layer = LinearLayer(num_classes)

    def call(self, inputs, training):
        inputs = self.linear_layer(inputs, training)
        inputs = tf.identity(inputs, name='logits_sup')
        return inputs

class Model(tf.keras.models.Model):
    #Resnet model with supervised layer

    def __init__(self, num_classes, **kwargs):
        super(Model, self).__init__(**kwargs)
        #resnet
        self.resnet_model = resnet.resnet(resnet_depth=FLAGS.resnet_depth, cifar_stem=FLAGS.image_size <=32)
        self._projection_head = ProjectionHead()
        self.supervised_head = SupervisedHead(num_classes)
    
    def __call__(self, inputs, training):
        features = inputs
        num_transforms = 1
        
        #split channels and optionally apply extra batched augmentation
        features_list = tf.split(features, num_or_size_splits=num_transforms, axis=-1)
        features = tf.concat(features_list, 0) #(num_transforms * bsz, h, w, c)

        #base network forward pass
        hiddens = self.resnet_model(features, training=training)

        #add heads
        supervised_head_inputs = self._projection_head(hiddens, training)

        supervised_head_outputs = self.supervised_head(supervised_head_inputs, training)
        return supervised_head_outputs