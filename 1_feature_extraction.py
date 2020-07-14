#from multiprocessing import Pool, Process
#from settings import *
import os
import numpy as np
from header_construction import *
#from feature_extraction import *
#import entropy

malware_exe_dir = "4500_malware_exe"
benign_exe_dir  = "39000_benign_exe"
csv_file_path   = "dataset_csv/1804_features.csv"
APIS_PATH       = "APIs.txt"

def get_byte_code(f):
    byte_code = f.read()

    return byte_code


def get_int_code(byte_code):   
    int_code = np.zeros((len(byte_code)), dtype=np.uint8)

    for i, byte in enumerate(byte_code):            
        int_code[i] = int(byte)

    return int_code
        

def get_byte_1gram(int_code):
    one_byte = np.zeros((256), dtype=np.uint32)

    for integer in int_code:
        one_byte[integer] += 1

    return one_byte


def get_byte_meta_data(byte_code):
    meta_data = []

    # file size
    file_size = len(byte_code)
    meta_data.append(file_size)

    # PE header start address (3C to 3F)
    start_address_hexs = byte_code[int(0x3C):int(0x40)]
    start_address_int  = 0
    for i, start_address_hex in enumerate(start_address_hexs):
        start_address_int += int(start_address_hex) * (256 ** i)    
    meta_data.append(start_address_int)    

    return meta_data


def get_quantile_diffs(block_entropys, threshold_list):
    quantile_diffs = []
    prev_quantile  = 0

    for threshold in threshold_list:
        now_quantile = float((block_entropys < threshold).sum()) / len(block_entropys)
        quantile_diffs.append(now_quantile - prev_quantile)

        prev_quantile = now_quantile

    return quantile_diffs

def get_statistic_values(block_entropys):
	mean     = np.mean(block_entropys)
	variance = np.var(block_entropys)
	median   = np.median(block_entropys)
	maximum  = np.max(block_entropys)
	minimum  = np.min(block_entropys)
	max_min  = maximum - minimum

	return [mean, variance, median, maximum, minimum, max_min]



def get_byte_entropy(int_code):
    int_code_len   = len(int_code)
    windows_size   = 10000
    stride_size    = 100
    block_size     = ((int_code_len - windows_size) // stride_size) + 1
    block_entropys = np.zeros((block_size))

    byte_count = np.zeros((256))
    for i in range(windows_size):
        byte_count[int_code[i]] += 1

    entropy = 0
    for i in range(len(byte_count)):
        if byte_count[i] > 0:
            entropy += -(byte_count[i] / windows_size) * np.log2(byte_count[i] / windows_size)

    block_entropys[0] = entropy

    for i in range(1, int_code_len-windows_size, 1):
        decrease_count = byte_count[int_code[i - 1]]
        increase_count = byte_count[int_code[i - 1 + windows_size]]

        byte_count[int_code[i - 1]] -= 1
        byte_count[int_code[i - 1 + windows_size]] += 1

        entropy -= -(decrease_count / windows_size) * np.log2(decrease_count / windows_size)

        if decrease_count > 1:
            entropy += -((decrease_count - 1) / windows_size) * np.log2((decrease_count - 1) / windows_size)

        if increase_count > 0:
            entropy -= -(increase_count / windows_size) * np.log2(increase_count / windows_size)

        entropy += -((increase_count + 1)) / windows_size * np.log2((increase_count + 1) / windows_size)

        if i % stride_size == 0:
            block_entropys[i // stride_size] = entropy


    quantile_diffs   = get_quantile_diffs(block_entropys, np.arange(0.2, 4.4, 0.2)) # 21 features
    statistic_values = get_statistic_values(block_entropys)                         #  6 features
   




def feature_extraction(malware_exe_dir, benign_exe_dir, csv_file_path):
    colnames = ['filename']
    # 6 hex dump-based features
    colnames += header_byte_1gram()
    colnames += header_byte_meta_data()
    colnames += header_byte_entropy()
    colnames += header_byte_img1()
    colnames += header_byte_img2()    
    colnames += header_byte_str_len()
    # 7 disassemble-based features
    colnames += header_asm_meta_data()
    colnames += header_asm_sym()
    colnames += header_asm_registers()
    colnames += header_asm_opcodes()
    colnames += header_asm_sections()
    colnames += header_asm_data_define()
    colnames += header_asm_apis(APIS_PATH)
    # benign or malware
    colnames += ['label']

    malware_exe_names = os.listdir(malware_exe_dir)
    print(malware_exe_names)

    for i, malware_exe_name in enumerate(malware_exe_names):    
        malware_exe_path = malware_exe_dir + "/" + malware_exe_name
        f = open(malware_exe_path, "rb")
        byte_code = get_byte_code(f)
        int_code  = get_int_code(byte_code)

        byte_oneg        =  get_byte_1gram(int_code) 
        byte_meta_data   =  get_byte_meta_data(byte_code)
        byte_entropy     =  get_byte_entropy(int_code)
        #byte_image1      = byte_image1(f) 
        #byte_image2      = byte_image2(f)
        #

        #byte_str_lengths = byte_string_lengths(f)

        #asm_meta_data           = asm_meta_data(malware_exe_path, f)
        #asm_symbols             = asm_symbols(f)
        #asm_registers           = asm_registers(f)
        #asm_opcodes             = asm_opcodes(f)
        #asm_sections, asm_names = asm_sections(f)
        #asm_data_defines        = asm_data_define(f)
        #asm_apis                = asm_APIs(f,defined_apis)


feature_extraction(malware_exe_dir, benign_exe_dir, csv_file_path)