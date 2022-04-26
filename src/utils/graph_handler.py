import tensorflow as tf

#################################################################################
# Saver - tensorflow mechanism for "re-running" code without having to re-run it
#################################################################################
class GraphHandler(object):
    def __init__(self, model, logging):
        self.model = model
        self.logging = logging
        self.saver = tf.compat.v1.train.Saver(max_to_keep=3)
        self.writer = None

    def initialize(self, sess, cfg):
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())
        if cfg.globals["load_model"]: #or cfg.mode != 'train_dann':
            self.restore(sess)
        if cfg.globals["mode"] == 'train':
            print('cfg.summary_dir =',cfg.summary_dir)
            self.writer = tf.compat.v1.summary.FileWriter(logdir=cfg.summary_dir, graph=tf.get_default_graph())
        self.cfg = cfg

    def add_summary(self, summary, global_step):
        self.logging.add()
        self.logging.add('saving summary...')
        self.writer.add_summary(summary, global_step)
        self.logging.done()

    def add_summaries(self, summaries, global_step):
        for summary in summaries:
            self.add_summary(summary, global_step)

    def save(self, sess, global_step = None):
        self.logging.add()
        self.logging.add('saving model to %s'% self.cfg.ckpt_path)
        self.saver.save(sess, self.cfg.ckpt_path, global_step)
        self.logging.done()

    def restore(self,sess):
        self.logging.add()
        # print(self.cfg.ckpt_dir)

        if self.cfg.load_step is None:
            if self.cfg.load_path is None:
                self.logging.add('trying to restore from dir %s' % self.cfg.ckpt_dir)
                latest_checkpoint_path = tf.train.latest_checkpoint(self.cfg.ckpt_dir)
            else:
                latest_checkpoint_path = self.cfg.load_path
        else:
            latest_checkpoint_path = self.cfg.ckpt_path+'-'+str(self.cfg.load_step)

        if latest_checkpoint_path is not None:
            self.logging.add('trying to restore from ckpt file %s' % latest_checkpoint_path)
            try:
                self.saver.restore(sess, latest_checkpoint_path)
                self.logging.add('success to restore')
            except tf.errors.NotFoundError:
                self.logging.add('failure to restore')
                if self.cfg.globals["mode"] != 'train': raise FileNotFoundError('canot find model file')
        else:
            self.logging.add('No check point file in dir %s '% self.cfg.ckpt_dir)
            if self.cfg.globals["mode"] != 'train': raise FileNotFoundError('canot find model file')

        self.logging.done()


