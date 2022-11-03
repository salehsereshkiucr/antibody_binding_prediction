import antibody_cnn as ac

classifier_type = 'cnn'
antibody = 'CTLA-4'
another_antibody = 'CTLA-4'

for i in range(10):
    print(ac.classification(classifier_type, antibody, another_antibody))
