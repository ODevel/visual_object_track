import os
import shutil
#import sys

#datas_path = sys.argv[1]
tests = [
'football/',
'biker/', 
'bird1/', 
'blurbody/', 
'blurcar2/', 
'bolt/', 
'cardark/',
'human3/',
'human6/',
'human9/',
'panda/',
'walking/',
'walking2/']

#print(datas_path)
for datas_path in tests:
    datas_path += '/' + datas_path +'/'
    if(not os.path.exists(os.path.join(datas_path, 'test'))):
        os.mkdir(datas_path + '/test')
    
    files_p = os.listdir(datas_path + '/p')
    for (i,f) in enumerate(files_p):
        if((i % 7) == 0):
            if(i == 0):
                files_p_test = [f]
            else:
                files_p_test.append(f)
            shutil.move(os.path.join(datas_path + '/p', f), os.path.join(datas_path, 'test'))

