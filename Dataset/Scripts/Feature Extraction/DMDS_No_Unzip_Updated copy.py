import numpy as np
import os
import re
import json
import hashlib
import logging
from sklearn.feature_extraction import FeatureHasher
import traceback
import requests

import zipfile
import time
'''
Run in directory with 2 folders. /Malicious Reports and /Benign Reports. all the cuckoo sandbox reports.
Reads all the .json files in the directory, does feature extraction on them 
and outputs a .npy file with the same name as input_file with the numpy array contained.

e.g report1.json will output report1.npy

'''
class FeatureType(object):
    ''' Base class from which each feature type may inherit '''

    name = ''
    dim = 0

    def __repr__(self):
        return '{}({})'.format(self.name, self.dim)

    def raw_features(self, input_dict):
        ''' Generate a JSON-able representation of the file '''
        raise (NotImplemented)

    def process_features(self, raw_obj):
        ''' Generate a feature vector from the raw features '''
        raise (NotImplemented)

    def feature_vector(self, input_dict):
        ''' Directly calculate the feature vector from the sample itself. This should only be implemented differently
        if there are significant speedups to be gained from combining the two functions. '''

        return self.process_raw_features(self.raw_features(input_dict))


class APIName(FeatureType):
    ''' api_name hash info '''

    name = 'api_name'
    dim = 8

    def __init__(self):
        super(FeatureType, self).__init__()
        self._name = re.compile('^[a-z]+|[A-Z][^A-Z]*')

    def raw_features(self, input_dict):
        """
        input_dict: string
        """
        tmp = self._name.findall(input_dict)
        hasher = FeatureHasher(self.dim, input_type="string").transform([tmp]).toarray()[0]
        return hasher

    def process_raw_features(self, raw_obj):
        return raw_obj


class APICategory(FeatureType):
    ''' api_category hash info '''
    
    name = 'api_category'
    dim = 4

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, input_dict):
        hasher = FeatureHasher(self.dim, input_type="string").transform([list(input_dict)]).toarray()[0]
        return hasher

    def process_raw_features(self, raw_obj):
        return raw_obj


class IntInfo(FeatureType):
    ''' int hash info '''

    name = 'int'
    dim = 16

    def __init__(self):
        super(FeatureType, self).__init__()

    def raw_features(self, input_dict):
        hasher = FeatureHasher(self.dim).transform([input_dict]).toarray()[0]
        return hasher

    def process_raw_features(self, raw_obj):
        return raw_obj


class PRUIInfo(FeatureType):
    ''' Path, Registry, Urls, IPs hash info '''

    name = 'prui'
    dim = 16 + 8 + 12 + 16 + 12

    def __init__(self):
        super(FeatureType, self).__init__()
        self._paths = re.compile('^c:\\\\', re.IGNORECASE)
        self._dlls = re.compile('.+\.dll$', re.IGNORECASE)
        self._urls = re.compile('^https?://(.+?)[/|\s|:]', re.IGNORECASE)
        self._registry = re.compile('^HKEY_')
        self._ips = re.compile('^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')

    def raw_features(self, input_dict):
        paths = np.zeros((16,), dtype=np.float32)
        dlls = np.zeros((8,), dtype=np.float32)
        registry = np.zeros((12,), dtype=np.float32)
        urls = np.zeros((16,), dtype=np.float32)
        ips = np.zeros((12,), dtype=np.float32)
        for str_name, str_value in input_dict.items():
            if self._dlls.match(str_value):
                tmp = re.split('//|\\\\|\.', str_value)[:-1]
                tmp = ['\\'.join(tmp[:i]) for i in range(1, len(tmp) + 1)]
                dlls += FeatureHasher(8, input_type="string").transform([tmp]).toarray()[0]
            if self._paths.match(str_value):
                tmp = re.split('//|\\\\|\.', str_value)[:-1]
                tmp = ['\\'.join(tmp[:i]) for i in range(1, len(tmp) + 1)]
                paths += FeatureHasher(16, input_type="string").transform([tmp]).toarray()[0]
            elif self._registry.match(str_value):
                tmp = str_value.split('\\')[:6]
                tmp = ['\\'.join(tmp[:i]) for i in range(1, len(tmp) + 1)]
                registry += FeatureHasher(12, input_type="string").transform([tmp]).toarray()[0]
            elif self._urls.match(str_value):
                tmp = self._urls.split(str_value + "/")[1]
                tmp = tmp.split('.')[::-1]
                tmp = ['.'.join(tmp[:i][::-1]) for i in range(1, len(tmp) + 1)]
                urls += FeatureHasher(16, input_type="string").transform([tmp]).toarray()[0]
            elif self._ips.match(str_value):
                tmp = str_value.split('.')
                tmp = ['.'.join(tmp[:i]) for i in range(1, len(tmp) + 1)]
                ips += FeatureHasher(12, input_type="string").transform([tmp]).toarray()[0]
        return np.hstack([paths, dlls, registry, urls, ips]).astype(np.float32)

    def process_raw_features(self, raw_obj):
        return raw_obj


class StringsInfo(FeatureType):
    ''' Other printable strings hash info '''

    name = 'strings'
    dim = 8

    def __init__(self):
        super(FeatureType, self).__init__()
        self._allstrings = re.compile(b'[\x20-\x7f]{5,}')
        self._paths = re.compile(b'c:\\\\', re.IGNORECASE)
        self._dlls = re.compile(b'\\.dll', re.IGNORECASE)
        self._urls = re.compile(b'https?://', re.IGNORECASE)
        self._registry = re.compile(b'HKEY_')
        self._mz = re.compile(b'MZ')
        self._ips = re.compile(b'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
        super(FeatureType, self).__init__()

    def raw_features(self, input_dict):
        bytez = '\x11'.join(input_dict.values()).encode('UTF-8', 'ignore')
        allstrings = self._allstrings.findall(bytez)
        if allstrings:
            # statistics about strings:
            string_lengths = [len(s) for s in allstrings]
            avlength = sum(string_lengths) / len(string_lengths)
            # map printable characters 0x20 - 0x7f to an int array consisting of 0-95, inclusive
            as_shifted_string = [b - ord(b'\x20') for b in b''.join(allstrings)]
            c = np.bincount(as_shifted_string, minlength=96)  # histogram count
            # distribution of characters in printable strings
            csum = c.sum()
            p = c.astype(np.float32) / csum
            wh = np.where(c)[0]
            H = np.sum(-p[wh] * np.log2(p[wh]))  # entropy
        else:
            avlength = 0
            c = np.zeros((96,), dtype=np.float32)
            H = 0
            csum = 0
        return {
            'numstrings': len(allstrings),
            'avlength': avlength,
            'printables': int(csum),
            'entropy': float(H),
            'paths': len(self._paths.findall(bytez)),
            'dlls': len(self._dlls.findall(bytez)),
            'urls': len(self._urls.findall(bytez)),
            'registry': len(self._registry.findall(bytez)),
            'ips': len(self._ips.findall(bytez)),
            'MZ': len(self._mz.findall(bytez))
        }

    def process_raw_features(self, raw_obj):
        return np.hstack([
            raw_obj['numstrings'], raw_obj['avlength'], raw_obj['printables'],
            raw_obj['entropy'], raw_obj['paths'], raw_obj['dlls'], raw_obj['urls'],
            raw_obj['registry'], raw_obj['ips'], raw_obj['MZ']
        ]).astype(np.float32)

class ReturnValue(FeatureType):
    ''' return_value hash info '''

    name = 'return_value'
    dim = 2

    def __init__(self):
        super(FeatureType, self).__init__()
        self._name = re.compile("\d+")

    def raw_features(self, return_value):
        """
        input_dict: string
        """
        return_value = str(return_value)
        hasher = FeatureHasher(self.dim, input_type="string").transform([list(return_value)]).toarray()[0]
        return hasher

    def process_raw_features(self, raw_obj):
        return raw_obj

class DMDS(object):

    def __init__(self, file_name, input_path, output_path, max_len, idx):
        logging.info('Generating vector for task %s' % idx)

        self.idx = idx
        self.behaviour_report = None
        self.nrdma_output = None
        self.max_len = max_len
        self.features = dict((fe.name, fe) for fe in
                             [APIName(), APICategory(), IntInfo(), PRUIInfo(), StringsInfo(), ReturnValue()])
        self.input_path = input_path
        self.output_path = output_path
        self.file_name = file_name
        self.infile = self.input_path.format(self.file_name)
        self.outfile = self.output_path.format(self.file_name)
        self.feature_vector = np.zeros((258,105))
        self.list_of_api_calls = ["GetTimeZoneInformation", "WSAAccept", "Process32Next", "GetLocalTime", "GetSystemWindowsDirectory", "NtDeleteKey", "NtOpenDirectoryObject", "NtCreateKey", "Connect", "NtCreateUserProcess", "__anomaly__", "NetGetJoinInformation", "RtlDispatchException", "OpenService", "FindResource", "recv", "NtTerminateThread", "ControlService", "WSAConnect", "ShellExecute", "GetKeyState", "RtlCreateUserProcess", "NtCreateSection", "GetComputerName", "RegQueryInfoKey", "NtClose", "RtlCompressBuffer", "CryptCreateHash", "GetVolumePathNamesForVolumeName", "GetFileSize", "PRF", "listen", "CoInitialize", "SetFileInformationByHandle", "NtQueryDirectoryFile", "recvfrom", "NtDuplicateObject", "bind", "GetFileAttributes", "NtMakeTemporaryObject", "SearchPath", "NtReplaceKey", "GetBestInterface", "InternetCloseHandle", "Module32First", "NtSuspendThread", "GetVolumeNameForVolumeMountPoint", "GetAsyncKeyState", "WSASend", "NtDelayExecution", "CryptGenKey", "WriteProcessMemory", "RtlAddVectoredExceptionHandler", "NtWriteFile", "NtResumeThread", "NtRenameKey", "RegDeleteKey", "InternetGetConnectedState", "CryptEncryptMessage", "InternetQueryOption", "NtLoadKey", "NtQueryFullAttributesFile", "NetShareEnum", "NtDeleteFile", "RemoveDirectory", "LdrGetDllHandle", "SetUnhandledExceptionFilter", "__missing__", "GetAdaptersInfo", "getaddrinfo", "gethostbyname", "NtUnmapViewOfSection", "GetSystemInfo", "LookupAccountSid", "OpenSCManager", "InternetSetOption", "connect", "NtGetContextThread", "GetTempPath", "CreateProcessInternal", "NtCreateDirectoryObject", "LdrGetProcedureAddress", "InternetOpen", "NtReadVirtualMemory", "Thread32First", "GetForegroundWindow", "RtlRemoveVectoredContinueHandler", "NtEnumerateKey", "NtCreateThread", "InternetSetStatusCallback", "ExitWindows", "CWindow_AddTimeoutCode", "UuidCreate", "NetUserGetLocalGroups", "HttpSendRequest", "RegDeleteValue", "DeleteService", "CryptProtectMemory", "EnumServicesStatus", "NetUserGetInfo", "NtMapViewOfSection", "NtCreateFile", "HttpOpenRequest", "NtQueryAttributesFile", "UnhookWindowsHook", "DnsQuery_UTF8", "NtTerminateProcess", "GetSystemTimeAsFileTime", "NtAllocateVirtualMemory", "RegEnumKey", "NtLoadKey2", "IsDebuggerPresent", "GetCursorPos", "NtProtectVirtualMemory", "RegCloseKey", "NtQueryValueKey", "socket", "ReadCabinetState", "CreateThread", "GetUserName", "NtOpenProcess", "NtSetValueKey", "GetVolumePathName", "CDocument_write", "CryptHashData", "InternetReadFile", "GetInterfaceInfo", "RtlDecompressFragment", "CryptDecodeMessage", "MoveFileWithProgress", "GetTickCount", "ObtainUserAgentString", "SHGetSpecialFolderLocation", "GetNativeSystemInfo", "SetErrorMode", "InternetWriteFile", "RegQueryValue", "CryptDecodeObject", "DeleteUrlCacheEntry", "HttpQueryInfo", "NtFreeVirtualMemory", "CoCreateInstance", "RegCreateKey", "LoadResource", "NtUnloadDriver", "NtQuerySystemTime", "FindFirstFile", "NtSetInformationFile", "send", "WSASocket", "CIFrameElement_CreateElement", "getsockname", "NtEnumerateValueKey", "NtQueryInformationFile", "Module32Next", "WSARecv", "SetFilePointer", "CertCreateCertificateContext", "GetFileType", "CertOpenStore", "OleInitialize", "NtQueueApcThread", "URLDownloadToFile", "InternetCrackUrl", "NtDeleteValueKey", "CScriptElement_put_src", "GetSystemTime", "CryptDecrypt", "CryptExportKey", "__exception__", "Process32First", "NtMakePermanentObject", "FindWindow", "SendNotifyMessage", "SetWindowsHook", "shutdown", "COleScript_Compile", "LoadString", "setsockopt", "NtQueryMultipleValueKey", "RegOpenKey", "RegEnumValue", "DeviceIoControl", "CertOpenSystemStore", "RtlCreateUserThread", "NtSaveKey", "CoInitializeSecurity", "Thread32Next", "RtlRemoveVectoredExceptionHandler", "GetSystemMetrics", "__process__", "LdrLoadDll", "GetShortPathName", "CryptProtectData", "sendto", "NtOpenSection", "NtOpenThread", "RegSetValue", "CryptUnprotectMemory", "CreateRemoteThread", "SizeofResource", "InternetOpenUrl", "CryptAcquireContext", "NtReadFile", "DrawText", "Ssl3GenerateKeyMaterial", "CreateDirectory", "CryptUnprotectData", "WriteConsole", "timeGetTime", "CryptDecryptMessage", "GetSystemDirectory", "MessageBoxTimeout", "NtCreateMutant", "GetDiskFreeSpace", "system", "CryptHashMessage", "SetEndOfFile", "WSAStartup", "TransmitFile", "WSARecvFrom", "LdrUnloadDll", "select", "NtOpenKey", "EnumWindows", "GetFileInformationByHandle", "CHyperlink_SetUrlComponent", "NtSetContextThread", "NtWriteVirtualMemory", "ioctlsocket", "CreateService", "CElement_put_innerHTML", "OutputDebugString", "accept", "RtlDecompressBuffer", "DeleteFile", "CopyFile", "CreateToolhelp32Snapshot", "GetAddrInfo", "NtCreateProcess", "NtDeviceIoControlFile", "RtlAddVectoredContinueHandler", "InternetConnect", "GetKeyboardState", "NtQueryKey", "NtLoadDriver", "SHGetFolderPath", "CertControlStore", "LookupPrivilegeValue", "SetFileAttributes", "DnsQuery_", "CryptEncrypt", "WSASendTo", "ReadProcessMemory", "closesocket", "StartService", "GetAdaptersAddresses", "NtOpenFile"]

    # Opens the cuckoo report at the specified location and loads it into a json object
    def parse(self):
        if not os.path.exists(self.infile):
            logging.warning("Behaviour report does not exist.")
            print(f"{self.infile} does not exist")
            return False
        if os.path.exists(self.outfile):
            logging.warning("Behaviour report already parsed.")
            return False
        try:
            json_data = open(self.infile, "r")
            self.behaviour_report = json.load(json_data)
            return True
        except Exception as e:
            logging.error('Could not parse the behaviour report. {%s}' % e)
            return False

    def write(self):
        outputfile = self.output_path.format(self.file_name)
        logging.info("Writing task %s report to: %s" % (self.idx, outputfile))
        np.save(outputfile, self.feature_vector)
        # print("Calling Save File Function!")
        # print(f"Outfile: {outputfile}")
        # print(f"Data being saved: {self.feature_vector.size}")
        return True

    

    def add_to_output(self, sample, api):
        if api in self.list_of_api_calls:
            
            indexs = self.list_of_api_calls.index(api)
            self.feature_vector[indexs][0:12] = sample[0:12]
            self.feature_vector[indexs][12:104] += sample[12:104]
            self.feature_vector[indexs][104] += 1
        #else:
            #print(f"280 api not found leh! {api}")
        return True  

    """
        if self.nrdma_output is None:
            self.nrdma_output = [sample]
        else:
            self.nrdma_output = np.append(self.nrdma_output, [sample], axis=0)
        return len(self.nrdma_output)
    """


    def convert_thread(self, pid, tid, api_calls):
        #previous_hashed = ""
        for call in api_calls:
            #i += 1
            if 'api' not in call:
                continue
            if call['api'][:2] == '__':
                continue
            if 'arguments' not in call:
                call['arguments'] = {}
            if 'category' not in call:
                call['category'] = ""
            if 'status' not in call:
                call['status'] = 0
            if 'return_value' not in call:
                call["return_value"] = 0
            arguments = call['arguments']
            category = call['category']
            api = self.remove_api_suffix(call['api'])
            return_value = call["return_value"]
            #call_sign = api + "-" + str(arguments)
            #current_hashed = hashlib.md5(call_sign.encode()).hexdigest()
            #if previous_hashed == current_hashed:
            #    continue
            #else:
            #    previous_hashed = current_hashed
            api_name_hashed = self.features['api_name'].feature_vector(api)
            api_category_hashed = self.features['api_category'].feature_vector(
                category)
            api_int_dict, api_str_dict = {}, {}
            for c_n, c_v in arguments.items():
                if isinstance(c_v, (list, dict, tuple)):
                    continue
                if isinstance(c_v, (int, float)):
                    api_int_dict[c_n] = np.log(np.abs(c_v) + 1)
                else:
                    if c_v[:2] == '0x':
                        continue
                    api_str_dict[c_n] = c_v
            try:
                api_int_hashed = self.features['int'].feature_vector(api_int_dict)
                api_prui_hashed = self.features['prui'].feature_vector(
                    api_str_dict)
                api_str_hashed = self.features['strings'].feature_vector(
                    api_str_dict)
                api_return_value_hashed = self.features["return_value"].feature_vector(return_value)
                hashed_feature = np.hstack(
                    [api_name_hashed, api_category_hashed, api_int_hashed, api_prui_hashed, api_str_hashed, api_return_value_hashed]).astype(
                    np.float32)
                self.add_to_output(hashed_feature, api)
            except Exception:
                print(traceback.format_exc())
                pass
        return True       

    #  Launch the conversion on all threads in the JSON
    def convert(self):
        processes = {}
        try:
            procs = self.behaviour_report['behavior']['processes']
            apistats = self.behaviour_report['behavior']['apistats']
            for proc in procs:
                process_id = proc['pid'] # Process ID: 452
                parent_id = proc['ppid'] # Parent ID: 356
                process_name = proc['process_name'] # Process Name: lsass.exe
            
                calls = proc['calls']
                #  Create a dictionnary of threads
                # The key is the n° of the thread
                # The content is all calls he makes


                '''
                
                Creates a dictionary called threads.

                After appending, it will be: {'2272' : 'data of the call' }
                '''
                threads = {}
                for call in calls:
                    # Added a if here to prevent trying to access non existent data
                    if(call):
                        thread_id = call['tid']
                    try:
                        threads[thread_id].append(call)
                        # print(f"Call is {call}")
                    except:
                        threads[thread_id] = []
                        threads[thread_id].append(call)
                # Create a dictionnary of process
                # The key is the id of the process
                processes[process_id] = {}
                processes[process_id]["parent_id"] = parent_id
                processes[process_id]["process_name"] = process_name
                processes[process_id]["threads"] = threads
        except Exception:
            print(traceback.format_exc())
        # For all processes...
        for p_id in processes:
            #  For each threads of those processes...
            for t_id in processes[p_id]["threads"]:
                # Convert the thread
                if(t_id):
                    self.convert_thread(p_id, t_id, processes[p_id]["threads"][t_id])
        return True

    def remove_api_suffix(self, api):
        suffix_to_remove = ['ExW', 'ExA', 'W', 'A', 'Ex']
        for item in suffix_to_remove:
            if re.search(item+'$', api):
                return re.sub(item+'$', '', api)
        return api

 

if __name__ == '__main__':

    # benign_files = os.listdir('C:/Users/naush/OneDrive - Singapore Institute Of Technology/Benign Reports')
    # benign_reports_path = [] 
    # benign_outputs_path = []

    # malicious_files = os.listdir('C:/Users/naush/OneDrive - Singapore Institute Of Technology/Malicious Reports')
    # malicious_reports_path = [] 
    # malicious_outputs_path = []

    # # LOGGING CODES HERE.
    # file_count = len(malicious_files) + len(benign_files)
    current_count = 0

    # for file in benign_files:
    #     if file.endswith('.json'):
    #         benign_reports_path.append(file)
    #         benign_outputs_path.append(file.replace('.json','.npy'))
    # benign_file_paths = zip(benign_reports_path,benign_outputs_path)
    # benign_file_paths = list(benign_file_paths)
    # for input_path, output_path in benign_file_paths:

        # Unzip the file at input path
        
    with (zipfile.ZipFile('E:\\43k_malign_dataset_reports-18k_ok.zip', 'r')) as zipper:
        file_count = len(zipper.infolist())

        print(f"FILECOUNT={file_count}")
        for file_info in zipper.infolist():
            if file_info.is_dir() or ".txt" in file_info.filename:
                continue
            input_file_name = file_info.filename

            output_file_name = file_info.filename.split("/")[-1].removesuffix('.json') + '.npy'
            if '-' in output_file_name:
                output_file_name = output_file_name.split('-')[0] + '.npy'
                print(f"Next: {output_file_name}")
            # print(f"fileInfo: {file_info}")
            # Extract that file and run dmds only if the output npy file is not already there. 
            if(not os.path.exists('malNpy/'+output_file_name)):
                zipper.extract(file_info, 'unzipped')

                dmds = DMDS(input_file_name, input_path='unzipped/' + input_file_name, output_path='malNpy/'+output_file_name,max_len=1000,idx = 40)
                try:
                    if dmds.parse() and dmds.convert():
                        dmds.write()
                        try:
                            # Sends a request to the server i setup. To Keep track of the progress in parsing the files. 
                            current_count += 1
                            status_message = f"Parsed{current_count}/{file_count}Files."
                            print(status_message)
                            # requests.get(f"http://172.174.247.34:8000/{status_message}", timeout=5)
                        except Exception as e:
                            print("Error connecting to VM.")
                    else:
                        print("dmds failed")
                    os.remove('unzipped/' + file_info.filename)
                except Exception as e:
                    # requests.get(f"http://172.174.247.34:8000/{status_message}", timeout=5)
                    print("Following Error caught and handled: %s", e)
            else:
                print("Skipped a file that has already been parsed.")
                current_count+=1

        # Update input path to the newly extracted file



    # dmds = DMDS(input_path, "C:/Users/naush/OneDrive - Singapore Institute Of Technology/Benign Reports/"+ input_path, "C:/Users/naush/OneDrive - Singapore Institute Of Technology/Benign Npy/" + output_path, 1000, 40)
    # if dmds.parse() and dmds.convert():
    #     dmds.write()
    #     try:
    #         # Sends a request to the server i setup. To Keep track of the progress in parsing the files. 
    #         current_count += 1
    #         status_message = f"Parsed{current_count}/{file_count}Files."
    #         requests.get(f"http://172.174.247.34:8000/{status_message}", timeout=5)
    #     except Exception as e:
    #         print("Error connecting to VM.")
    # else:
    #     logging.warning("Failed in parsing report.")

    
    # for file in malicious_files:
    #     #print(file)
    #     if file.endswith('.json'):
    #        malicious_reports_path.append(file)
    #        malicious_outputs_path.append(file.replace('.json','.npy'))
    # malicious_file_paths = zip(malicious_reports_path,malicious_outputs_path)
    # malicious_file_paths = list(malicious_file_paths)
    # for input_path, output_path in malicious_file_paths:
    #     dmds = DMDS(input_path, "C:/Users/naush/OneDrive - Singapore Institute Of Technology/Malicious Reports/"+ input_path, "C:/Users/naush/OneDrive - Singapore Institute Of Technology/Malicious Npy/" + output_path, 1000, 40)
    #     if dmds.parse() and dmds.convert():
    #         try:
    #             # Sends a request to the server i setup. To Keep track of the progress in parsing the files. 
    #             current_count += 1
    #             status_message = f"Parsed{current_count}/{file_count}Files."
    #             r = requests.get(f"http://172.174.247.34:8000/{status_message}",timeout=5)
    #         except Exception as e:
    #             print("Error connecting to VM")

    #     else:
    #         logging.warning("Failed in parsing report.")