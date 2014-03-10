from subprocess import check_call

for i in xrange(9):
    print ('detection for testeye1_' + str(i))
    check_call('python detectEyeShape.py testeye1_' + str(i)+'.png', shell=True)
    print ('detection for testeye2_' + str(i))
    check_call('python detectEyeShape.py testeye2_' + str(i)+'.png', shell=True)
    
