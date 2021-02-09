from absl import app
from absl import flags
from absl import logging
import os
import math
import json
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

import data_dogs_eval as data_lib
import model_eval as model_lib
import modelsave


FLAGS = flags.FLAGS
########################## Flags ##################################

############ Used in resnet_eval
flags.DEFINE_boolean('global_bn', False, 'Whether to aggregate BN statistics across distributed cores.')
flags.DEFINE_float('batch_norm_decay', 0.9, 'Batch norm decay parameter.')

############ Used in model_eval
flags.DEFINE_float('weight_decay', 1e-6, 'Amount of weight decay to use.')
flags.DEFINE_integer('num_proj_layers', 3, 'Number of non-linear head layers.')
flags.DEFINE_integer('proj_out_dim', 128, 'Number of head projection dimension.')
flags.DEFINE_integer('ft_proj_selector', 0,'Which layer of the projection head to use during fine-tuning. '
        '0 means no projection head, and -1 means the final layer.')
flags.DEFINE_integer('resnet_depth', 18, 'Depth of ResNet.')
flags.DEFINE_integer('image_size', 100, 'Input image size.')

############  Used in pretrain (here)
flags.DEFINE_string('data_dir', None, 'Directory where the images are')
flags.DEFINE_integer('percentage', 10, 'Percentage of dataset to use in finetuning')
flags.DEFINE_integer('eval_steps', 0,'Number of steps to eval for. If not provided, evals over entire dataset.')
flags.DEFINE_string('model_dir', None,'Model directory for training.')
flags.DEFINE_integer('eval_batch_size', 256,'Batch size for eval.')
flags.DEFINE_integer('train_epochs', 300, 'Number of epochs to train for.')
flags.DEFINE_integer('train_batch_size', 512, 'Batch size for training.')
flags.DEFINE_integer('train_steps', 0, 'Number of steps to train for. If provided, overrides train_epochs.')
############  modelsave
flags.DEFINE_integer('keep_hub_module_max', 1,'Maximum number of Hub modules to keep.')

######################## End Flags ################################

def perform_evaluation(model, builder, eval_steps, ckpt, strategy):
    #perform evaluation
    ds = data_lib.build_distributed_dataset(builder, FLAGS.eval_batch_size, strategy)

    summary_writer = tf.summary.create_file_writer(FLAGS.model_dir)

    #Build metrics
    with strategy.scope():
        regularization_loss = tf.keras.metrics.Mean('eval/regularization_loss')
        label_top_1_accuracy = tf.keras.metrics.Accuracy('eval/label_top_1_accuracy')
        all_metrics = [regularization_loss, label_top_1_accuracy]

        #Restore checkpoint
        logging.info('Restoring from %s', ckpt)
        checkpoint = tf.train.Checkpoint(model=model, global_step=tf.Variable(0, dtype=tf.int64))
        checkpoint.restore(ckpt).expect_partial()
        global_step = checkpoint.global_step
        logging.info('Performing eval at step %d', global_step.numpy())

    def single_step(features, labels):
        supervised_head_outputs = model(features, training=False)
        #assert supervised_head_outputs is not None
        outputs = supervised_head_outputs
        l = labels['labels']
        ########## Update metrics ##############################
        label_top_1_accuracy.update_state(tf.argmax(l, 1), tf.argmax(outputs, axis=1))
        ##################################################
        reg_loss = model_lib.add_weight_decay(model)
        regularization_loss.update_state(reg_loss)

    with strategy.scope():

        @tf.function
        def run_single_step(iterator):
            images, labels = next(iterator)
            features, labels = images, {'labels': labels}
            strategy.run(single_step, (features, labels))

        iterator = iter(ds)
        for i in range(eval_steps):
            run_single_step(iterator)
            logging.info('Completed eval for %d/%d steps', i+1, eval_steps)
        logging.info('Finished eval for %s', ckpt)
    
    #Write summaries
    cur_step = global_step.numpy()
    logging.info('Writing summaries for %d step', cur_step)
    with summary_writer.as_default():
        for metric in all_metrics:
            metric_value = metric.result().numpy().astype(float)
            logging.info('Step: [%d] %s = %f', cur_step, metric.name, metric_value)
            tf.summary.scalar(metric.name, metric_value, step=cur_step)
        summary_writer.flush()

    #Record results as JSON.
    result_json_path = os.path.join(FLAGS.model_dir, 'result.json')
    result = {metric.name: metric.result().numpy() for metric in all_metrics}
    result['global_step'] = global_step.numpy()
    logging.info(result)
    with tf.io.gfile.GFile(result_json_path, 'w') as f:
        json.dump({k: float(v) for k, v in result.items()}, f)
    result_json_path = os.path.join(FLAGS.model_dir, 'result_%d.json'%result['global_step'])
    with tf.io.gfile.GFile(result_json_path, 'w') as f:
        json.dump({k: float(v) for k, v in result.items()}, f)
    flag_json_path = os.path.join(FLAGS.model_dir, 'flags.json')
    with tf.io.gfile.GFile(flag_json_path, 'w') as f:
        serializable_flags = {}
        for key, val in FLAGS.flag_values_dict().items():
            #Some flag value types e.g. datetime.timedelta are not json serializable,
            #filter those out
            if json_serializable(val):
                serializable_flags[key] = val
            
        json.dump(serializable_flags, f)

    modelsave.save(model, global_step=result['global_step'])
    
    return result

def json_serializable(val):
    try:
        json.dumps(val)
        return True
    except TypeError:
        return False


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments')

    logging.info('Using Cats_vs_Dogs dataset from TensorFlow Datasets')
    builder = tfds.builder('cats_vs_dogs', data_dir=FLAGS.data_dir)
    builder.download_and_prepare()
    num_train_examples = builder.info.splits['train[:{}%]'.format(FLAGS.percentage)].num_examples
    num_classes = builder.info.features['label'].num_classes
    num_eval_examples = builder.info.splits['train[-{}%:]'.format(FLAGS.percentage)].num_examples

    train_steps = FLAGS.train_steps or (num_train_examples * FLAGS.train_epochs // FLAGS.train_batch_size + 1)
    eval_steps = FLAGS.eval_steps or int(math.ceil(num_eval_examples / FLAGS.eval_batch_size))

    logging.info('# train examples: %d', num_train_examples)
    logging.info('# train_steps: %d', train_steps)
    logging.info('# eval examples: %d', num_eval_examples)
    logging.info('# eval steps: %d', eval_steps)


    strategy = tf.distribute.MirroredStrategy()
    logging.info('Running using MirroredStrategy on %d replicas', strategy.num_replicas_in_sync)

    #instanciating model
    with strategy.scope():
        model = model_lib.Model(num_classes)
    
    for ckpt in tf.train.checkpoints_iterator(FLAGS.model_dir, min_interval_secs = 15):
        result = perform_evaluation(model, builder, eval_steps, ckpt, strategy)

        if result['global_step'] >= train_steps:
            logging.info('Eval complete. Exiting')
            return

if __name__ == '__main__':
    tf.compat.v1.enable_v2_behavior()
    #necessary for GPU // For outside compilation of summaries on TPU.
    tf.config.set_soft_device_placement(True)
    app.run(main)