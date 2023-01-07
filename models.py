class Model:
    display_name: str
    file_prefix: str

    def __init__(self, display_name, file_prefix):
        self.display_name = display_name
        self.file_prefix = file_prefix


MODELS = [
    Model(display_name='TC_n', file_prefix='tc_n'),
    Model(display_name='TC_f', file_prefix='tc_f'),
    Model(display_name='TC_nf', file_prefix='tc_nf'),
    Model(display_name='TC_fp', file_prefix='tc_fp'),
    Model(display_name='TC_nfp', file_prefix='tc_nfp'),
    Model(display_name='TC_jn', file_prefix='tc_jn'),
    Model(display_name='TC_jf', file_prefix='tc_jf'),
    Model(display_name='TC_jfp', file_prefix='tc_jfp'),
]

NORMALIZATION_METHODS = ['l1', 'l2', 'max']
