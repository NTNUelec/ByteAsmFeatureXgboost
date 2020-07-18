#from multiprocessing import Pool, Process
#from settings import *
import os
import numpy as np
from header_construction import *
import mahotas
from numba import jit, prange
import r2pipe
#from pwn import *
#from feature_extraction import *
#import entropy

malware_exe_dir = "4500_malware_exe"
benign_exe_dir  = "39000_benign_exe"
csv_file_path   = "dataset_csv/1804_features.csv"
APIS_PATH       = "APIs.txt"



def get_int_code(exe_path):   
    int_code = np.fromfile(exe_path, dtype=np.uint8)

    return int_code
        
@jit(parallel=True)
def get_byte_1gram(int_code):
    one_byte = np.zeros((256))

    for i in prange(len(int_code)):
        value = int_code[i]
        one_byte[value] += 1

    return one_byte


def get_byte_meta_data(int_code):
    meta_data = []

    # file size
    file_size = len(int_code)
    meta_data.append(file_size)

    # PE header start address (3C to 3F)
    start_address_hexs = int_code[int(0x3C):int(0x40)]
    start_address_int  = 0

    for i, start_address_hex in enumerate(start_address_hexs):
        start_address_int += start_address_hex * (256 ** i)    
    meta_data.append(start_address_int)    

    return meta_data

@jit(parallel=True)
def get_quantile_diffs(block_entropys, threshold_list):
    quantile_diffs = np.zeros((len(threshold_list)))
    prev_quantile  = 0

    for i in prange(len(threshold_list)):
        threshold    = threshold_list[i]
        now_quantile = float((block_entropys < threshold).sum()) / len(block_entropys)        
        quantile_diffs[i] = now_quantile - prev_quantile

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


def get_entropy_diffs(block_entropys):
    entropy_diffs = np.diff(block_entropys)

    return entropy_diffs


def get_entropy_diffs_quantile_diffs(block_entropys, threshold_list):
    entropy_diffs = get_entropy_diffs(block_entropys)
    entropy_diffs_quantile_diffs = get_quantile_diffs(entropy_diffs, threshold_list)

    return  entropy_diffs_quantile_diffs


def get_entropy_diffs_statistic_values(block_entropys):
    entropy_diffs = get_entropy_diffs(block_entropys)
    entropy_diffs_statistic_values = get_statistic_values(entropy_diffs)

    return entropy_diffs_statistic_values

#@jit(parallel=True)
def get_zone_quantile_diffs(block_entropys, zone_num, threshold_list):
    zone_quantile_diffs = np.zeros((zone_num * len(threshold_list)))
    zone_size = len(block_entropys) // zone_num

    for i in prange(zone_num):
        zone_entropys                  = block_entropys[i * zone_size: (i + 1) * zone_size]
        zone_quantile_diff             = get_quantile_diffs(zone_entropys, threshold_list)
        start                          = i * len(zone_quantile_diff)
        end                            = (i + 1) * len(zone_quantile_diff)
        zone_quantile_diffs[start:end] = zone_quantile_diff    

    return zone_quantile_diffs


#@jit(parallel=True)
def get_zone_statistic_values(block_entropys, zone_num):
    zone_statistic_values = np.zeros((zone_num * 6))
    zone_size = len(block_entropys) // zone_num

    for i in prange(zone_num):
        zone_entropys                    = block_entropys[i * zone_size: (i + 1) * zone_size]
        zone_statistic_value             = get_statistic_values(zone_entropys)
        start                            = i * len(zone_statistic_value)
        end                              = (i + 1) * len(zone_statistic_value)
        zone_statistic_values[start:end] = zone_statistic_value

    return zone_statistic_values


def get_percentile(block_entropys, num, diffs=False):
    sorted_block_entropys = np.sort(block_entropys)
    step = np.floor(float(len(sorted_block_entropys)) // num) - 1
    percentile_values = [sorted_block_entropys[int(step) * i] for i in range(num)]

    if diffs:
        percentile_values_diffs = np.ediff1d(percentile_values, to_begin=percentile_values[0])
        return percentile_values_diffs        
    else:
        return percentile_values

#@jit(parallel=True)
def get_byte_entropy(int_code):
    int_code_len   = len(int_code)
    windows_size   = 10000
    stride_size    = 100
    zone_num       = 4 
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


    quantile_diffs                 = get_quantile_diffs(block_entropys, np.arange(0.2, 4.4, 0.2))                  #  21 features
    statistic_values               = get_statistic_values(block_entropys)                                          #   6 features
    entropy_diffs_quantile_diffs   = get_entropy_diffs_quantile_diffs(block_entropys, np.arange(-0.1, 0.11, 0.01)) #  21 features   
    entropy_diffs_statistic_values = get_entropy_diffs_statistic_values(block_entropys)                            #   6 features
    zone_quantile_diffs            = get_zone_quantile_diffs(block_entropys, zone_num, np.arange(0.2, 4.4, 0.2))   #  84 features
    zone_statistic_values          = get_zone_statistic_values(block_entropys, zone_num)                           #  24 features
    percentile_values              = get_percentile(block_entropys, 20, diffs=False)                               #  20 features    
    percentile_values_diffs        = get_percentile(block_entropys, 20, diffs=True)                                #  20 features
    total_entropy                  = np.mean(block_entropys)                                                       #   1 features              
  
    byte_entropy = []
    byte_entropy.extend(quantile_diffs)
    byte_entropy.extend(statistic_values)
    byte_entropy.extend(entropy_diffs_quantile_diffs)
    byte_entropy.extend(entropy_diffs_statistic_values)
    byte_entropy.extend(zone_quantile_diffs)
    byte_entropy.extend(zone_statistic_values)
    byte_entropy.extend(percentile_values)
    byte_entropy.extend(percentile_values_diffs)
    byte_entropy.append(total_entropy)  

    return byte_entropy


def byte_make_image(int_code):
    size = int(np.sqrt(len(int_code)))
    img_array = int_code[:size ** 2]   
    img_array = np.reshape(img_array, (size, size))
    
    return img_array


def get_byte_image1(int_code):
    img1_feature      = []
    img_array         = byte_make_image(int_code)
    haralick_features = mahotas.features.haralick(img_array)

    for i in range(len(haralick_features)):
        for j in range(len(haralick_features[0])):            
            img1_feature.append(haralick_features[i][j])

    return img1_feature


def get_byte_image2(int_code):
    img_array    = byte_make_image(int_code)
    lbp_features = mahotas.features.lbp(img_array, 10, 10, ignore_zeros=False)
    img2_feature = lbp_features.tolist()    

    return img2_feature


def get_str_len_count(int_code, str_start_end, ascii_list):
    max_str_num = str_start_end.shape[0]
   
    now_end = len(int_code) - 1
    pre_end = -1
    state   = 0
   
    min_length  = 4
    count_1     = 0
    count_2     = 0
    count_3     = 0
    now_str_num = 0

    while now_end >= 1:
        # 0x00 is a string end
        if int_code[now_end] == 0:
            if (state == 1):               
                diff = pre_end - now_end - 1

                if (diff >= min_length):
                    str_start_end[now_str_num, 0] = now_end + 1
                    str_start_end[now_str_num, 1] = pre_end
                    now_str_num += 1

                elif (diff == 1):
                    count_1 += 1

                elif (diff == 2):
                    count_2 += 1

                elif (diff == 3):
                    count_3 += 1
                    
            state   = 1
            pre_end = now_end

        elif int_code[now_end] not in ascii_list:                
            if (state == 1):
                state = 0
                diff  = pre_end - now_end - 1

                if (diff >= min_length):
                    str_start_end[now_str_num, 0] = now_end + 1
                    str_start_end[now_str_num, 1] = pre_end
                    now_str_num += 1

                elif (diff == 1):
                    count_1 += 1

                elif (diff == 2):
                    count_2 += 1

                elif (diff == 3):
                    count_3 += 1
        
        now_end -= 1

        if now_str_num == max_str_num:
            break

    return now_str_num, count_1, count_2, count_3


def get_strings(int_code):
    name = ''
    ascii_list = np.array([i for i in range(32, 127)] + [13, 10])
    ascii_list.sort()
   
    str_start_end = np.zeros((15000, 2), dtype=np.int64)    
    now_str_num, count_1, count_2, count_3 = get_str_len_count(int_code, str_start_end, ascii_list)
    
    string_total_len = np.sum(str_start_end[:, 1] - str_start_end[:, 0]) + count_1 + count_2 + count_3
    string_ratio     = float(string_total_len) / len(int_code)

    strings = []
    for i in range(now_str_num):
        strings.extend([''.join([chr(x) for x in int_code[str_start_end[i, 0]: str_start_end[i, 1]]])])   

    return name, strings, count_1, count_2, count_3, string_total_len, string_ratio


def extract_length(name, strings, count_1, count_2, count_3, string_total_len, string_ratio):    
    # min string length is 0, and max is 10000
    min_len, max_len = 0, 10000   
    len_arrays = np.array([len(y) for y in strings] + [min_len] + [max_len])   
    bincounts  = np.bincount(len_arrays)
    bincounts[0]     -= 1
    bincounts[10000] -= 1   
   
    counts_0_10       = np.sum(bincounts[0:10]) + count_1 + count_2 + count_3
    counts_10_30      = np.sum(bincounts[10:30])
    counts_30_60      = np.sum(bincounts[30:60])
    counts_60_90      = np.sum(bincounts[60:90]) 
    counts_90_100     = np.sum(bincounts[90:100]) 
    counts_100_150    = np.sum(bincounts[100:150])
    counts_150_250    = np.sum(bincounts[150:250])
    counts_250_400    = np.sum(bincounts[250:450])
    counts_400_600    = np.sum(bincounts[400:600])
    counts_600_900    = np.sum(bincounts[600:900])
    counts_900_1300   = np.sum(bincounts[900:1300])
    counts_1300_2000  = np.sum(bincounts[1300:2000])
    counts_2000_3000  = np.sum(bincounts[2000:3000])
    counts_3000_6000  = np.sum(bincounts[3000:6000])
    counts_6000_15000 = np.sum(bincounts[6000:15000])


    feature = []
    feature.extend([count_1, count_2, count_3])
    feature.extend([bincounts[i] for i in range(4, 100)])
    feature.extend([counts_0_10,
                    counts_10_30,
                    counts_30_60,
                    counts_60_90,
                    counts_90_100,
                    counts_100_150,
                    counts_150_250,
                    counts_250_400,
                    counts_400_600,
                    counts_600_900,
                    counts_900_1300,
                    counts_1300_2000,
                    counts_2000_3000,
                    counts_3000_6000,
                    counts_6000_15000,
                    string_total_len,
                    string_ratio
                    ])                          
                   
    return feature

def get_byte_string_lengths(int_code):
    name, strings, count_1, count_2, count_3, string_total_len, string_ratio = get_strings(int_code)
    byte_string_lengths = extract_length(name, strings, count_1, count_2, count_3, string_total_len, string_ratio)

    return byte_string_lengths


def get_asm_code(exe_path):
    file = r2pipe.open(exe_path)
    asm_code = file.cmd('pd $s')
    asm_code = np.array(asm_code.split("\n"))

    return asm_code


def get_asm_meta_data(asm_code):   
    meta_data = []

    # file size
    file_size = len(asm_code)
    meta_data.append(file_size)

    # row number
    row_num = 0
    for row in asm_code:
        row_num += 1
    meta_data.append(row_num)

    return meta_data


def get_asm_symbols(asm_code):
    symbols = np.zeros((7))

    for row in asm_code:
        if '*' in row:
            symbols[0] += 1
        if '-' in row:
            symbols[1] += 1
        if '+' in row:
            symbols[2] += 1
        if '[' in row:
            symbols[3] += 1
        if ']' in row:
            symbols[4] += 1
        if '@' in row:
            symbols[5] += 1
        if '?' in row:
            symbols[6] += 1

    return symbols


def get_asm_registers(asm_code):
    registers = ['edx','esi','es','fs','ds','ss','gs','cs','ah','al',
                 'ax','bh','bl','bx','ch','cl','cx','dh','dl','dx',
                 'eax','ebp','ebx','ecx','edi','esp']

    registers_values = np.zeros((len(registers))) 

    for row in asm_code:
        parts =   row.replace(',', ' ')
        parts = parts.replace('+', ' ')
        parts = parts.replace('*', ' ')
        parts = parts.replace('[', ' ')
        parts = parts.replace(']', ' ')
        parts = parts.replace('-', ' ')
        parts = parts.split()

        for register in registers:
            registers_values[registers.index(register)] += parts.count(register)
            
    return registers_values


def get_asm_opcodes(asm_code):
    opcodes = ['add','al','bt','call','cdq','cld','cli','cmc','cmp','const','cwd','daa','db'
                ,'dd','dec','dw','endp','ends','faddp','fchs','fdiv','fdivp','fdivr','fild'
                ,'fistp','fld','fstcw','fstcwimul','fstp','fword','fxch','imul','in','inc'
                ,'ins','int','jb','je','jg','jge','jl','jmp','jnb','jno','jnz','jo','jz'
                ,'lea','loope','mov','movzx','mul','near','neg','not','or','out','outs'
                ,'pop','popf','proc','push','pushf','rcl','rcr','rdtsc','rep','ret','retn'
                ,'rol','ror','sal','sar','sbb','scas','setb','setle','setnle','setnz'
                ,'setz','shl','shld','shr','sidt','stc','std','sti','stos','sub','test'
                ,'wait','xchg','xor']

    opcodes_values = np.zeros((len(opcodes))) 

    for row in asm_code:
        parts = row.split()

        for opcode in opcodes:
            if opcode in parts:
                opcodes_values[opcodes.index(opcode)] += 1
                break

    return opcodes_values


def read_api_file(APIS_PATH):
    file = open(APIS_PATH, "r")
    content = file.read()
    apis = np.array(content.split(","))

    return apis


def get_asm_APIs(asm_code, apis):
    apis_values = np.zeros((len(apis)))
    for row in asm_code:      
        for i in range(len(apis)):
            if apis[i] in row:
                apis_values[i] += 1 
                break

    return apis_values


def get_asm_sections(asm_code):
    section_names = []
    for row in asm_code:
        section_name = [row[0: np.core.defchararray.index(row, ':')]]
        if section_name != 'HEADER':
            section_names += section_name

    known_sections = ['.text', '.data', '.bss', '.rdata', '.edata', '.idata', '.rsrc', '.tls', '.reloc']
    sections_values = [0]*24
    unknown_sections = []
    unknown_lines = 0
    number_of_sections = len(section_names)

    for section in section_names:

        if section in known_sections:
            section_index = known_sections.index(section)
            sections_values[section_index] += 1
        else:
            unknown_sections.append(section)
            unknown_lines += 1

    uni_section_names_len = len(np.unique(section_names))
    uni_unknown_section_names_len = len(np.unique(unknown_sections))
    uni_known_section_names_len = 0
    for i in range(0,8):
        if sections_values[i] != 0:
            uni_known_section_names_len += 1

    sections_values[9] = uni_section_names_len
    sections_values[10] = uni_unknown_section_names_len
    sections_values[11] = unknown_lines

    for i in range(0,8):
        sections_values[i + 12] = float(sections_values[i])/ number_of_sections

    sections_values[21] = float(uni_known_section_names_len) / uni_section_names_len
    sections_values[22] = float(uni_unknown_section_names_len) / uni_section_names_len
    sections_values[23] = float(unknown_lines) / number_of_sections

    return sections_values, section_names

def asm_misc(asm_code):

    keywords = ['Virtual','Offset','loc','Import','Imports','var','Forwarder','UINT','LONG','BOOL','WORD','BYTES','large','short','dd','db','dw','XREF','ptr','DATA','FUNCTION','extrn','byte','word','dword','char','DWORD','stdcall','arg','locret','asc','align','WinMain','unk','cookie','off','nullsub','DllEntryPoint','System32','dll','CHUNK','BASS','HMENU','DLL','LPWSTR','void','HRESULT','HDC','LRESULT','HANDLE','HWND','LPSTR','int','HLOCAL','FARPROC','ATOM','HMODULE','WPARAM','HGLOBAL','entry','rva','COLLAPSED','config','exe','Software','CurrentVersion','__imp_','INT_PTR','UINT_PTR','---Seperator','PCCTL_CONTEXT','__IMPORT_','INTERNET_STATUS_CALLBACK','.rdata:','.data:','.text:','case','installdir','market','microsoft','policies','proc','scrollwindow','search','trap','visualc','___security_cookie','assume','callvirtualalloc','exportedentry','hardware','hkey_current_user','hkey_local_machine','sp-analysisfailed','unableto']

    keywords_values = [0]*len(keywords)
    for row in asm_code:
        #parts = row.replace(',',' ').replace('+',' ').replace('*',' ').replace('[',' ').replace(']',' ') \
        #            .replace('-',' ').split()
        for i in range(len(keywords)):
            if keywords[i] in row:
                keywords_values[i] += 1 #parts.count(opcode)
                break
    return keywords_values


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
    
    defined_apis = read_api_file(APIS_PATH)

    for i, malware_exe_name in enumerate(malware_exe_names):    
        malware_exe_path = malware_exe_dir + "/" + malware_exe_name

        """ dump-based features """     
        """
        int_code         = get_int_code(malware_exe_path)

        byte_oneg        = get_byte_1gram(int_code)          #   2 features
        byte_meta_data   = get_byte_meta_data(int_code)      # 256 features
        byte_entropy     = get_byte_entropy(int_code)        # 203 features        
        byte_image1      = get_byte_image1(int_code)         #  52 features     
        byte_image2      = get_byte_image2(int_code)         # 108 features 
        byte_str_lengths = get_byte_string_lengths(int_code) # 116 features
        """


      
        """ disassemble-based features """
        #file      = open(malware_exe_path, "rb")        
        #asm_code  = disasm(byte_code, arch = 'i386')     

        
        
        asm_code       = get_asm_code(exe_path)
        asm_meta_data          = get_asm_meta_data(asm_code)
        asm_symbols             = get_asm_symbols(asm_code)
        asm_registers           = get_asm_registers(asm_code)
        asm_opcodes             = get_asm_opcodes(asm_code)
        asm_apis                = get_asm_APIs(asm_code, defined_apis)        
        asm_sections, asm_names = get_asm_sections(asm_code)

        #asm_data_defines        = asm_data_define(f)
        #


feature_extraction(malware_exe_dir, benign_exe_dir, csv_file_path)