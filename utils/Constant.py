import os

# TRAIN CONSTANTS
TIME_INTERVAL = ['15min', '30min', '60min']
MIN_15 = ['AT', 'BE', 'DE', 'DE_50hertz', 'DE_LU', 'DE_amprion', 'DE_tennet', 'HU', 'LU', 'NL']
MIN_30 = ['CY', 'GB_GBN', 'GB_UKM', 'IE']
MIN_60 = ['AT', 'BE', 'DE', 'DE_50hertz', 'DE_LU', 'DE_amprion', 'DE_tennet', 'HU',
          'LU', 'NL', 'BG', 'CY', 'GB_GBN', 'GB_UKM', 'IE', 'CH', 'CZ', 'DK', 'DK_1', 'DK_2',
          'EE', 'ES', 'FI', 'FR', 'GR', 'HR', 'IE_sem', 'IT', 'LT', 'LV', 'NO', 'NO_1',
          'NO_2', 'SK']
TRAIN_PATH = os.path.join('Train', 'train_data')
TRAIN_15 = os.path.join(TRAIN_PATH, '15min')
TRAIN_30 = os.path.join(TRAIN_PATH, '30min')
TRAIN_60 = os.path.join(TRAIN_PATH, '60min')
WEIGHTS_PATH = os.path.join('Models', 'autoencoder_weights')
