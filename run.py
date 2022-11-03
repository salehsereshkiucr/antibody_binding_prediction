from tensorflow.python.ops.metrics_impl import recall

import antibody_cnn as ac

classifier_type = 'rf'
antibody = 'CTLA-4'
another_antibody = 'CTLA-4'

for i in range(10):
    accuracy, f1, precision, recalll = ac.classification(classifier_type, antibody, another_antibody)
    with open("classifier_acc.txt", "a") as file_object:
        file_object.write('%s\t%s\t%s\t%s\t%s\t%s\t%s' %(classifier_type, antibody, another_antibody, str(accuracy),
                                                     str(f1), str(precision), str(recall)))
        file_object.write("\n")
    print('%s\t%s\t%s\t%s\t%s\t%s\t%s' %(classifier_type, antibody, another_antibody, str(accuracy),
                                                     str(f1), str(precision), str(recall)))


classifier_type = 'rf'
antibody = 'PD-1'
another_antibody = 'PD-1'

for i in range(10):
    accuracy, f1, precision, recalll = ac.classification(classifier_type, antibody, another_antibody)
    with open("classifier_acc.txt", "a") as file_object:
        file_object.write('%s\t%s\t%s\t%s\t%s\t%s\t%s' %(classifier_type, antibody, another_antibody, str(accuracy),
                                                     str(f1), str(precision), str(recall)))
        file_object.write("\n")
    print('%s\t%s\t%s\t%s\t%s\t%s\t%s' %(classifier_type, antibody, another_antibody, str(accuracy),
                                                     str(f1), str(precision), str(recall)))
