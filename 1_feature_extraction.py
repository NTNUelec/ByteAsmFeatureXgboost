from multiprocessing import Pool, Process
from settings import *
import os

from byte_code_extraction_facade import byte_extraction
from asm_extraction_facade import asm_extraction
from feature_selection_facade import feature_fusion
from classification_facade import classification

malware_exe_dir = "4500_malware_exe"
benign_exe_dir  = "39000_benign_exe"
csv_file_path   = "dataset_csv/1804_features.csv"

def feature_extraction(malware_exe_dir, benign_exe_dir, csv_file_path):

    colnames = ['filename']
    # 6 hex dump-based features
    colnames += header_byte_1gram()
    colnames += header_byte_meta_data()
    colnames += header_byte_img1()
    colnames += header_byte_img2()
    colnames += header_byte_entropy()
    colnames += header_byte_str_len()
    # 7 disassemble-based features
    colnames += header_asm_meta_data()
    colnames += header_asm_sym()
    colnames += header_asm_registers()
    colnames += header_asm_opcodes()
    colnames += header_asm_sections()
    colnames += header_asm_data_define()
    colnames += header_asm_apis()
    # benign or malware
    colnames += ['label']


    malware_exe_names = os.listdir(malware_exe_dir)

    for i, malware_exe_name in enumerate(malware_exe_names):    	
    	malware_exe_path = malware_exe_dir + "/" + malware_exe_name
    	f = open(malware_exe_path, "r")    

        byte_oneg        = byte_1gram(f)      
        byte_meta_data   = byte_meta_data(malware_exe_path, f)
        byte_image1      = byte_image1(f) 
        byte_image2      = byte_image2(f)
        byte_entropy     = byte_entropy(f)
        byte_str_lengths = byte_string_lengths(f)

        asm_meta_data           = asm_meta_data(malware_exe_path, f)
        asm_symbols             = asm_symbols(f)
        asm_registers           = asm_registers(f)
        asm_opcodes             = asm_opcodes(f)
        asm_sections, asm_names = asm_sections(f)
        asm_data_defines        = asm_data_define(f)
        asm_apis                = asm_APIs(f,defined_apis)


