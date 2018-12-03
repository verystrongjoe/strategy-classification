raw_files_dir = 'raw_files/'
raw_file_ext = '.csv'
pickle_dir = 'prepocessed_data/'
pickle_file = 'data.pickle'

nn_hidden_size = 100
nn_input_size = 26
nn_output_size = 11

use_cuda = False

n_epochs = 10
n_batch_size = 40

n_train_files = 50

# training data meta information
l_columns_categorical = ['emGas', 'myBaseEmGas', 'emIsClose', 'checkEmBase', 'ememyBaseIsVisible']
l_columns_with_min_max = [
    ['drInMyBase','0','4'],
    ['allLing','0','30'],
    ['emBaseLing','0','30'],
    ['emFrontLing','0','30'],
    ['myBaseLing','0','30'],
    ['myFrontLing','0','30'],
    ['otherLing','0','30'],
    ['allHat','0','5'],
    ['allHatCom','0','5'],
    ['allHatUnCom','0','5'],
    ['emBaseHat','0','5'],
    ['emBaseHatCom','0','5'],
    ['emBaseHatUnCom','0','5'],
    ['emFrontHat','0','5'],
    ['emFrontHatCom','0','5'],
    ['emFrontHatUnCom','0','5'],
    ['nearMeHat','0','1'],
    ['nearMeHatCom','0','1'],
    ['nearMeHatUnCom','0','1'],
    ['hatBefore3M','0','3'],
    ['frontHatHP','0','1250']
]


mnist_data = 'pytorch/data'
n_threads = 1