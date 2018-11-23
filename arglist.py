raw_files_dir = 'raw_files/'
raw_file_ext = '.csv'
pickle_dir = 'prepocessed_data/'
pickle_file = 'data.pickle'

nn_hidden_size = 24
nn_input_size = 26
nn_output_size = 11

no_cuda = True

epochs = 10


# training data meta information
l_columns_categorical = ['EmGas', 'MyBaseEmGas', 'EmIsClose', 'CheckEmBase']
l_columns_with_min_max = [
    ['DrInMyBase','0','4'],
    ['AllLing','0','30'],
    ['EmBaseLing','0','30'],
    ['EmFrontLing','0','30'],
    ['MyBaseLing','0','30'],
    ['MyFrontLing','0','30'],
    ['OtherLing','0','30'],
    ['AllHat','0','5'],
    ['AllHatCom','0','5'],
    ['AllHatUnCom','0','5'],
    ['EmBaseHat','0','5'],
    ['EmBaseHatCom','0','5'],
    ['EmBaseHatUnCom','0','5'],
    ['EnFrontHat','0','5'],
    ['EnFrontHatCom','0','5'],
    ['EnFrontHatUnCom','0','5'],
    ['NearMeHat','0','1'],
    ['NearMeHatCom','0','1'],
    ['NearMeHatUnCom','0','1'],
    ['HatBefore3M','0','3'],
    ['FrontHatHP','0','1250']
]