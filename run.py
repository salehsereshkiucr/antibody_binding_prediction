import antibody_cnn as ac

classifier_type = 'cnn'
antibody = 'CTLA-4'
#antibody = 'PD-1'
another_antibody = 'CTLA-4'
encode_method='onehot'

for i in range(10):
    accuracy, f1, precision, encode_mat = ac.classification(classifier_type, antibody, another_antibody, encode_method='blosum')
    with open("classifier_acc.txt", "a") as file_object:
        file_object.write('%s,%s,%s,%s,%s,%s,%s' %(classifier_type, antibody, another_antibody, str(accuracy),
                                                     str(f1), str(precision), str(encode_mat)))
        file_object.write("\n")
    print('%s\t%s\t%s\t%s\t%s\t%s\t%s' %(classifier_type, antibody, another_antibody, str(accuracy),
                                                     str(f1), str(precision), str(encode_mat)))


classifier_type = 'cnn'
antibody = 'CTLA-4'
another_antibody = 'CTLA-4'

for i in range(10):
    accuracy, f1, precision, encode_mat = ac.classification(classifier_type, antibody, another_antibody, encode_method='blosum')
    with open("classifier_acc.txt", "a") as file_object:
        file_object.write('%s,%s,%s,%s,%s,%s,%s' %(classifier_type, antibody, another_antibody, str(accuracy),
                                                     str(f1), str(precision), str(encode_mat)))
        file_object.write("\n")
    print('%s\t%s\t%s\t%s\t%s\t%s\t%s' %(classifier_type, antibody, another_antibody, str(accuracy),
                                                     str(f1), str(precision), str(encode_mat)))

