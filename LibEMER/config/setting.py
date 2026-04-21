
class Setting:
    def __init__(self, dataset, dataset_path, pass_band, extract_bands,time_window, overlap, sample_length, stride,bio_length,bio_stride,seed,feature_type,use_multimodal = False, TnF=False,only_seg=False, extract_bio = False,
                cross_trail='true',eog_bands =None, emg_bands =None,gsr_bands = None,bvp_bands =None,resp_bands = None,temp_bands = None,ecg_bands =None,
                experiment_mode="sub_dependent", train_part=None, eog_clean=True,metrics=None, normalize=False, save_data=True, 
                split_type="kfold", fold_num=5, fold_shuffle=True, front=9,subject_divide = False, subject_front=12, test_size=0.2, val_size = 0.2,
                sessions=None, pr=None, sr=None, bounds=None,onehot=False, label_used=None):
        # random seed
        self.seed = seed

        #about multimodal EEGbased emotion recognition
        self.use_multimodal = use_multimodal

        # dataset setting

        self.dataset = dataset
        self.dataset_path = dataset_path
        
        # preprocess setting

        # Data at indices 0 and 1 represent the lower and higher thresholds of bandpass filtering
        self.pass_band = pass_band
        # Two-dimensional array, with each element at an index representing the range of each frequency band
        self.extract_bands = extract_bands 
        self.eog_bands = eog_bands
        self.emg_bands = emg_bands
        self.gsr_bands = gsr_bands
        self.bvp_bands = bvp_bands
        self.resp_bands = resp_bands
        self.temp_bands = temp_bands
        self.ecg_bands = ecg_bands
        self.bio_length = bio_length
        self.bio_stride = bio_stride

        # The size of the time window during preprocessing, in num of data points
        self.time_window = time_window
        # the length of overlap for each preprocessing window
        self.overlap = overlap
        # The length of sample sequences input to the model at once
        self.sample_length = sample_length
        # the stride of a sliding window for data extraction
        self.stride = stride
        # Feature type of EEG signals
        self.feature_type = feature_type
        # whether remove the eye movement interference
        self.eog_clean = eog_clean
        # whether normalize
        self.normalize = normalize
        # whether save_data
        self.save_data = save_data

        self.TnF = TnF
        self.only_seg = only_seg
        self.extract_bio = extract_bio
        # train_test_setting

        # whether use cross trial setting
        self.cross_trail = cross_trail
        # sub_dependent or sub_independent or cross_session
        self.experiment_mode = experiment_mode
        # how to partition a dataset
        self.split_type = split_type
        # according to the split type, choose which part is used as the training set or testing set
        self.fold_num = fold_num
        self.fold_shuffle = fold_shuffle
        self.front = front
        self.subject_divide = subject_divide
        self.subject_front = subject_front
        self.test_size = test_size
        self.val_size = val_size
        self.sessions = sessions
        self.pr = pr
        self.sr = sr

        self.bounds = bounds
        self.onehot = onehot
        self.label_used = label_used



def resolve_effective_experiment_mode(args):
    experiment_mode = getattr(args, "experiment_mode", None)
    setting_name = getattr(args, "setting", None)
    if setting_name is None:
        return experiment_mode

    for mode in ("sub_independent", "sub_dependent", "cross_session"):
        if mode in setting_name:
            return mode
    return experiment_mode


def resolve_effective_split_type(args):
    split_type = getattr(args, "split_type", None)
    setting_name = getattr(args, "setting", None)
    if setting_name is None:
        return split_type

    if "train_val_test" in setting_name:
        return "train_val_test"
    if "leave_one_out" in setting_name:
        return "leave_one_out"
    if "front_back" in setting_name:
        return "front_back"
    if "kfold" in setting_name or "5fold" in setting_name or "10fold" in setting_name:
        return "kfold"
    return split_type


def set_setting_by_args(args):
    if args.dataset_path is None:
        print("Please set the dataset path")
    if args.dataset is None:
        print("Please select the dataset to train")
    return Setting( dataset=args.dataset, use_multimodal= args.use_multimodal,dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, eog_bands=None, emg_bands=None,gsr_bands=None,bvp_bands=None,resp_bands=None,temp_bands=None,ecg_bands=None, 
                   time_window=args.time_window, overlap=args.overlap,sample_length=args.sample_length, stride=args.stride, bio_length= args.bio_length,
                   bio_stride= args.bio_stride,seed=args.seed, feature_type=args.feature_type, TnF=args.TnF, only_seg=args.only_seg,extract_bio = args.extract_bio, 
                   cross_trail=args.cross_trail, experiment_mode=args.experiment_mode,metrics=args.metrics, normalize=args.normalize, split_type=args.split_type, 
                   fold_num=args.fold_num,fold_shuffle=args.fold_shuffle, front=args.front, subject_divide=args.subject_divide,
                   subject_front= args.subject_front,sessions=args.sessions, pr=args.pr, sr=args.sr,
                   bounds=args.bounds, onehot=args.onehot, label_used=args.label_used)


def seed_sub_dependent_front_back_setting(args):
    if not args.dataset.startswith('seed'):
        print('not using SEED dataset, please check your setting')
        exit(1)
    print("Using Default SEED subject dependent experiment mode,\n"
          "the first 9 trails for each subject were used as a training set and the last 6 as a test set")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, experiment_mode="sub_dependent", normalize=args.normalize,
                   split_type='front_back', front=9, sessions=args.sessions, pr=args.pr, sr=args.sr, onehot=args.onehot,
                   label_used=args.label_used)

def seed_sub_dependent_train_val_test_setting(args):
    if not args.dataset.startswith('seed'):
        print('not using SEED dataset, please check your setting')
        exit(1)
    print("Using Seed subject dependent train val test experiment mode, \n"
          "For each subject, nine random trails were used as training set, three random trails were used as verification"
          " set, last three trails were used as test, we choose best results in verification set and test results in test")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, experiment_mode="sub_dependent", normalize=args.normalize,
                   split_type='train_val_test', test_size=0.2, val_size=0.2, sessions=args.sessions, pr=args.pr, sr=args.sr, onehot=args.onehot,
                   label_used=args.label_used)
def seed_multimodal_sub_dependent_train_val_test_setting(args):
    if not args.dataset.startswith('seed'):
        print('not using SEED dataset, please check your setting')
        exit(1)
    print("Using Seed subject dependent train val test experiment mode, \n"
          "For each subject, nine random trails were used as training set, three random trails were used as verification"
          " set, last three trails were used as test, we choose best results in verification set and test results in test")
    return Setting(dataset = args.dataset, use_multimodal=args.use_multimodal, dataset_path=args.dataset_path,pass_band=[args.low_pass, args.high_pass],
                   extract_bands = None, time_window = args.time_window, overlap = args.overlap,TnF = args.TnF,
                   sample_length = args.sample_length, stride = args.stride, bio_length=args.bio_length, bio_stride = args.bio_stride,
                   seed = args.seed, feature_type = args.feature_type,only_seg = args.only_seg, extract_bio = args.extract_bio,experiment_mode = "sub_dependent", normalize = args.normalize,
                   split_type = 'train_val_test', test_size = 0.2, val_size = 0.2, sessions = args.sessions, pr = args.pr, sr = args.sr, onehot=args.onehot,label_used = args.label_used
                    )
def seedv_sub_dependent_train_val_test_setting(args):
    if not args.dataset.startswith('seedv'):
        print('not using SEED V dataset, please check your setting')
        exit(1)
    print("Using SeedV subject dependent train val test experiment mode, \n"
          "For each subject, nine random trails were used as training set, three random trails were used as verification"
          " set, last three trails were used as test, we choose best results in verification set and test results in test")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, experiment_mode="sub_dependent", normalize=args.normalize,
                   split_type='train_val_test', test_size=0.2, val_size=0.2, sessions=[1,2,3], pr=args.pr,
                   sr=args.sr, onehot=args.onehot,
                   label_used=args.label_used)

def seedv_multimodal_sub_dependent_train_val_test_setting(args):
    if not args.dataset.startswith('seedv'):
        print('not using SeedV dataset, please check your setting')
        exit(1)
    print("Using SeedV subject dependent train val test experiment mode, \n"
          "For each subject, nine random trails were used as training set, three random trails were used as verification"
          " set, last three trails were used as test, we choose best results in verification set and test results in test")
    return Setting(dataset = args.dataset, use_multimodal=args.use_multimodal, dataset_path=args.dataset_path,pass_band=[args.low_pass, args.high_pass],
                   extract_bands = None, time_window = args.time_window, overlap = args.overlap,TnF = args.TnF,
                   sample_length = args.sample_length, stride = args.stride, bio_length=args.bio_length, bio_stride = args.bio_stride,
                   seed = args.seed, feature_type = args.feature_type,only_seg = args.only_seg, extract_bio = args.extract_bio, experiment_mode = "sub_dependent", normalize = args.normalize,
                   split_type = 'train_val_test', test_size = 0.2, val_size = 0.2, sessions = [1,2,3], pr = args.pr, sr = args.sr, onehot=args.onehot,label_used=args.label_used)

def seedv_sub_dependent_train_val_test_mean_setting(args):
    if not args.dataset.startswith('seedv'):
        print('not using SEED V dataset, please check your setting')
        exit(1)
    print("Using SeedV subject dependent train val test experiment mode, \n"
          "For each subject, five random trails were used as training set, five random trails were used as verification"
          " set, last five trails were used as test, we choose best results in verification set and test results in test")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, experiment_mode="sub_dependent", normalize=args.normalize,
                   split_type='train_val_test', test_size=0.34, val_size=0.34, sessions=[1,2,3], pr=args.pr,
                   sr=args.sr, onehot=args.onehot,
                   label_used=args.label_used)
def seedv_sub_independent_train_val_test_setting(args):
    if not args.dataset.startswith('seedv'):
        print('not using SEED V dataset, please check your setting')
        exit(1)
    print("Using SeedV subject independent train val test experiment mode, \n"
          "For each subject, 10 random subjects were used as training set, 3 random trails were used as verification"
          " set, last 3 trails were used as test, we choose best results in verification set and test results in test")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, experiment_mode="sub_independent", normalize=args.normalize,
                   split_type='train_val_test', test_size=0.2, val_size=0.2,
                   sessions=[1,2,3] if args.sessions is None else args.sessions,
                   pr=args.pr, sr=args.sr, onehot=args.onehot, label_used=args.label_used)

def seedv_multimodal_sub_independent_train_val_test_setting(args):
    if not args.dataset.startswith('seedv'):
        print('not using SEEDV dataset, please check your setting')
        exit(1)
    print("Using SeedV subject independent train val test experiment mode, \n"
          "For each experiment, 10 random subjects were used as training set, 3 random subjects were used as verification"
          " set, last 3 subjects were used as test, we choose best results in verification set and test results in test")
    return Setting(dataset = args.dataset, use_multimodal=args.use_multimodal, dataset_path=args.dataset_path,pass_band=[args.low_pass, args.high_pass],
                   extract_bands = None, time_window = args.time_window, overlap = args.overlap,TnF = args.TnF,
                   sample_length = args.sample_length, stride = args.stride, bio_length=args.bio_length, bio_stride = args.bio_stride,
                   seed = args.seed, feature_type = args.feature_type,only_seg = args.only_seg, extract_bio = args.extract_bio, experiment_mode = "sub_independent", normalize = args.normalize,
                   split_type = 'train_val_test', test_size = 0.2, val_size = 0.2, sessions = [1,2,3]if args.sessions is None else args.sessions, 
                   pr = args.pr, sr = args.sr, onehot=args.onehot,label_used=args.label_used)  

def seed_sub_dependent_5fold_setting(args):
    if not args.dataset.startswith('seed'):
        print('not using SEED dataset, please check your setting')
        exit(1)
    print("Using Default SEED subject dependent experiment mode,\n"
          "Using a 5-fold cross-validation, three test sets are grouped in the Order of trail")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, cross_trail=args.cross_trail, experiment_mode="sub_dependent",
                   normalize=args.normalize, split_type='kfold', fold_num=5, fold_shuffle=False, sessions=args.sessions,
                   pr=args.pr, sr=args.sr, onehot=args.onehot, label_used=args.label_used)


def seed_sub_independent_leave_one_out_setting(args):
    if not args.dataset.startswith('seed'):
        print('not using SEED dataset, please check your setting')
        exit(1)
    print("Using Default SEED subject independent early stopping experiment mode,\n"
          "Using the leave one out method, all samples of 15 trails for 1 subject were split "
          "into all samples as a test set, and all samples of 15 trails for 14 other round "
          "were split into all samples as a training set, cycle 15 times to report average results")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, experiment_mode="sub_independent", normalize=args.normalize,
                   split_type='leave_one_out', sessions=[1] if args.sessions is None else args.sessions,
                   pr=args.pr, sr=args.sr, onehot=args.onehot, label_used=args.label_used)

def seed_sub_independent_train_val_test_setting(args):
    if not args.dataset.startswith('seed'):
        print('not using SEED dataset, please check your setting')
        exit(1)
    print("Using train_val_test SEED subject independent experiment mode,\n"
          "The random nine subjects' data are taken as training set, random three subjects' data are taken as "
          "validation set, random three subject's data are taken as test set. We choose the best results in validation set,"
          "and test it in test set"
          )
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, experiment_mode="sub_independent", normalize=args.normalize,
                   split_type='train_val_test', test_size=0.2, val_size=0.2, sessions=[1] if args.sessions is None else args.sessions,
                   pr=args.pr, sr=args.sr, onehot=args.onehot, label_used=args.label_used)

def seed_multimodal_sub_independent_train_val_test_setting(args):
    if not args.dataset.startswith('seed'):
        print('not using SEED dataset, please check your setting')
        exit(1)  
    print("Using train_val_test SEED subject independent experiment mode,\n"
          "The random nine subjects' data are taken as training set, random three subjects' data are taken as "
          "validation set, random three subject's data are taken as test set. We choose the best results in validation set,"
          "and test it in test set"
          )
    return Setting(dataset = args.dataset, use_multimodal=args.use_multimodal, dataset_path=args.dataset_path,pass_band=[args.low_pass, args.high_pass],
                   extract_bands = None, time_window = args.time_window, overlap = args.overlap,TnF = args.TnF,
                   sample_length = args.sample_length, stride = args.stride, bio_length=args.bio_length, bio_stride = args.bio_stride,
                   seed = args.seed, feature_type = args.feature_type,only_seg = args.only_seg, extract_bio = args.extract_bio, experiment_mode = "sub_independent", normalize = args.normalize,
                   split_type = 'train_val_test', test_size = 0.2, val_size = 0.2, sessions = [1,2,3] if args.sessions is None else args.sessions, 
                   pr = args.pr, sr = args.sr, onehot=args.onehot,label_used=args.label_used)


def deap_sub_independent_train_val_test_setting(args):
    if not args.dataset.startswith('deap'):
        print('not using deap dataset, please check your setting')
        exit(1)
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, experiment_mode="sub_independent", normalize=args.normalize,
                   split_type='train_val_test', test_size=0.2, val_size=0.2, sessions=args.sessions, pr=args.pr, sr=args.sr,
                   onehot=args.onehot, bounds=args.bounds,
                   label_used=args.label_used)

def deap_multimodal_sub_independent_train_val_test_setting(args):
    if not args.dataset.startswith('deap'):
        print('not using deap dataset, please check your setting')
        exit(1)  
    print('Using Default DEAP multimodal sub independent train_val_test experiment mode,\n')
    return Setting(dataset = args.dataset, use_multimodal=args.use_multimodal, dataset_path=args.dataset_path,pass_band=[args.low_pass, args.high_pass],
                   extract_bands = None, time_window = args.time_window, overlap = args.overlap,TnF = args.TnF,
                   sample_length = args.sample_length, stride = args.stride, bio_length=args.bio_length, bio_stride = args.bio_stride,
                   seed = args.seed, feature_type = args.feature_type,only_seg = args.only_seg, extract_bio = args.extract_bio, experiment_mode = "sub_independent", normalize = args.normalize,
                   split_type = 'train_val_test', test_size = 0.2, val_size = 0.2, sessions = args.sessions, pr = args.pr, sr = args.sr, onehot=args.onehot,bounds = args.bounds,label_used=args.label_used)     

def deap_sub_dependent_train_val_test_setting(args):
    if not args.dataset.startswith('deap'):
        print('not using deap dataset, please check your setting')
        exit(1)
    print("Using deap subject dependent train_val_test experiment mode")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, experiment_mode="sub_dependent", normalize=args.normalize,
                   split_type='train_val_test', test_size=0.2, val_size=0.2, sessions=args.sessions, pr=args.pr, sr=args.sr,
                   onehot=args.onehot,bounds=args.bounds,
                   label_used=args.label_used)

def deap_multimodal_sub_dependent_train_val_test_setting(args):
    if not args.dataset.startswith('deap'):
        print('not using DEAP dataset, please check your setting')
        exit(1)
    print("Using Default DEAP sub dependent train_val_test experiment mode,\n")
    return Setting(dataset = args.dataset, use_multimodal=args.use_multimodal, dataset_path=args.dataset_path,pass_band=[args.low_pass, args.high_pass],
                   extract_bands = None, time_window = args.time_window, overlap = args.overlap,TnF = args.TnF,
                   sample_length = args.sample_length, stride = args.stride, bio_length=args.bio_length, bio_stride = args.bio_stride,
                   seed = args.seed, feature_type = args.feature_type,only_seg = args.only_seg, extract_bio = args.extract_bio, experiment_mode = "sub_dependent", normalize = args.normalize,
                   split_type = 'train_val_test', test_size = 0.2, val_size = 0.2, sessions = args.sessions, pr = args.pr, sr = args.sr, onehot=args.onehot,bounds = args.bounds,label_used=args.label_used)

def seed_cross_session_setting(args):
    if not args.dataset.startswith('seed'):
        print('not using SEED dataset, please check your setting')
        exit(1)
    print("Using Default SEED cross session experiment mode,\n"
          "Three sessions of data, one as the test dataset")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=None, time_window=args.time_window, overlap=args.overlap,
                   sample_length=args.sample_length, stride=args.stride, seed=args.seed, feature_type=args.feature_type,
                   only_seg=args.only_seg, experiment_mode="cross_session", normalize=args.normalize,
                   split_type='leave_one_out', sessions=args.sessions, pr=args.pr, sr=args.sr, onehot=args.onehot,
                   label_used=args.label_used)

def deap_sub_independent_leave_one_out_setting(args):
    if not args.dataset.startswith('deap'):
        print('not using DEAP dataset, please check your setting')
        exit(1)
    print("Using Default DEAP sub independent experiment mode,\n")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=[[4, 7], [8, 10], [8, 12], [13, 30], [30, 47]], time_window=args.time_window,
                   overlap=args.overlap, sample_length=args.sample_length, stride=args.stride, seed=args.seed,
                   feature_type=args.feature_type, only_seg=args.only_seg, experiment_mode="sub_independent",
                   normalize=args.normalize, split_type='leave_one_out', pr=args.pr, sr=args.sr, bounds=args.bounds,
                   onehot=args.onehot, label_used=args.label_used)

def deap_sub_dependent_10fold_setting(args):
    if not args.dataset.startswith('deap'):
        print('not using DEAP dataset, please check your setting')
        exit(1)
    print("Using Default DEAP sub independent experiment mode,\n")
    return Setting(dataset=args.dataset, dataset_path=args.dataset_path, pass_band=[args.low_pass, args.high_pass],
                   extract_bands=[[4, 7], [8, 10], [8, 12], [13, 30], [30, 47]], time_window=args.time_window,
                   overlap=args.overlap, sample_length=args.sample_length, stride=args.stride, seed=args.seed,
                   feature_type=args.feature_type, only_seg=args.only_seg, experiment_mode="sub_dependent",
                   normalize=args.normalize, cross_trail=args.cross_trail, split_type='kfold', fold_num=10, pr=args.pr, sr=args.sr, bounds=args.bounds,
                   onehot=args.onehot, label_used=args.label_used)

preset_setting = {
    "seed_sub_dependent_train_val_test_setting": seed_sub_dependent_train_val_test_setting,
    "seed_sub_independent_train_val_test_setting": seed_sub_independent_train_val_test_setting,
    "seedv_sub_dependent_train_val_test_mean_setting":seedv_sub_dependent_train_val_test_mean_setting,
    "seedv_sub_independent_train_val_test_setting":seedv_sub_independent_train_val_test_setting,
    "deap_sub_dependent_train_val_test_setting" : deap_sub_dependent_train_val_test_setting,
    "deap_sub_independent_train_val_test_setting" : deap_sub_independent_train_val_test_setting,
    # ***********************************************************************
    "seed_sub_dependent_5fold_setting": seed_sub_dependent_5fold_setting,
    "seed_sub_dependent_front_back_setting": seed_sub_dependent_front_back_setting,
    "seedv_sub_dependent_train_val_test_setting": seedv_sub_dependent_train_val_test_setting,
    "seed_sub_independent_leave_one_out_setting": seed_sub_independent_leave_one_out_setting,
    "seed_cross_session_setting": seed_cross_session_setting,
    "deap_sub_independent_leave_one_out_setting": deap_sub_independent_leave_one_out_setting,
    "deap_sub_dependent_10fold_setting": deap_sub_dependent_10fold_setting,
    #*******************************************************************************************
    "seed_multimodal_sub_dependent_train_val_test_setting": seed_multimodal_sub_dependent_train_val_test_setting,
    "seed_multimodal_sub_independent_train_val_test_setting": seed_multimodal_sub_independent_train_val_test_setting,
    "seedv_multimodal_sub_dependent_train_val_test_setting": seedv_multimodal_sub_dependent_train_val_test_setting,
    "seedv_multimodal_sub_independent_train_val_test_setting": seedv_multimodal_sub_independent_train_val_test_setting,
    "deap_multimodal_sub_dependent_train_val_test_setting": deap_multimodal_sub_dependent_train_val_test_setting,
    "deap_multimodal_sub_independent_train_val_test_setting": deap_multimodal_sub_independent_train_val_test_setting,

    None: set_setting_by_args
}
