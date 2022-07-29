import train_app_b
import train_app_a
import train_app_b_tuning
import train_app_c
import run_app_a
import run_app_b
import run_app_b_vid
import run_app_c
import apply_pca

#########################################################################
# CLI Class: Provides an interactive interface to the user to run the app
#########################################################################
class CLI :
    def __init__(self):
        print('Welcome to Visual Object Tracking :')

    def print_usage(self):
        usage = '''
----------------------------------
1) Press '1' to try simple detection (APP A)
2) Press '2' to try CNN based detection + tracking (APP B) & hyper-tuning.
3) Press '3' to try OpenCV based tracking (APP C)
4) Press '4' to run PCA before data training
5) Press 'q' to exit
        '''
        print(usage)

    def print_test_names(self) :
        names = ['biker', 'bird1', 'blurbody', 'blurcar2', 'bolt', 'cardark', 'football', 'human3', 'human6', 'human9', 'panda', 'walking', 'walking2']
        for n in names:
            print('    - ', n)

    def handle_app_a(self):
        usage = '''
APP A (Simple Detection):
-------------------------
1) Press '1' to try training of object types
2) Press '2' to run a test detection
3) Press '0' to return to main menu
        '''
        print(usage)
        key = input()
        while key != '0':
            if(key == '1'):
                train_app_a.train()
            elif(key == '2'):
                print('Give 3 test names:')
                self.print_test_names()
                tests = ['a'] * 3
                tests[0] = input()
                tests[1] = input()
                tests[2] = input()
                run_app_a.run(tests)
            else:
                print('Wrong input entered. Try again')
            print(usage)
            key = input()


    def handle_app_b(self):
        usage = '''
APP B (Detection + Tracking):
----------------------------
1) Press '1' to train using CNN based position annotation
2) Press '2' to try APP B on a sample image
3) Press '3' to try APP B on sample video clip with both approaches combined
4) Press '4' to train with hypertuning of CNN layers, activation functions and feature map
5) Press '0' to return to main menu
'''
        print(usage)
        key = input()
        while key != '0':
            if(key == '1'):
                print('Provide test name to train:')
                self.print_test_names()
                test = input()
                train_app_b.train(test)
            elif(key=='2'):
                print('Provide test name to run:')
                self.print_test_names()
                test = input()
                run_app_b.run(test)
            elif(key=='3'):
                print('Enter test name to run the clip: ')
                self.print_test_names()
                test = input()
                run_app_b_vid.run(test)
            elif(key=='4'):
                print('Enter test name to run the clip: ')
                self.print_test_names()
                test = input()
                train_app_b_tuning.train(test)
            else:
                print('Invalid input entered. Try again.')
            print(usage)
            key=input()


    def handle_app_c(self):
        usage = '''
APP C (OpenCV based Detection):
-------------------------------
(Please note that this approach uses HUD based cascade approach.)
1) Press '1' to train the trainer.yml 
2) Press '2' to try a sample video clip
3) Press '0' to go to main menu.
'''
        print(usage)
        key = input()
        while key != '0':
            if(key == '1'):
                print('Please enter a test case name: ')
                self.print_test_names()
                test = input()
                train_app_c.train(test)
            elif(key == '2'):
                print('Please enter a test case name: ')
                self.print_test_names()
                test = input()
                run_app_c.run(test)
            else:
                print('Wrong input entered')
            print(usage)
            key = input()

    def handle_pca(self):
        usage = '''
Enter test name or 0 to return to main menu:
'''
        print(usage)
        self.print_test_names()
        key = input()
        while key != '0':
            apply_pca.apply_pca(key)
            print(usage)
            self.print_test_names()
            key = input()

################################################################################
# Entry point for the application
################################################################################
if __name__ == '__main__':
    cli = CLI()
    cli.print_usage()
    key = input()
    while key != 'q':
        if(key == '1'):
            cli.handle_app_a()
        elif(key == '2'):
            cli.handle_app_b()
        elif(key == '3'):
            cli.handle_app_c()
        elif(key == '4'):
            cli.handle_pca()
        else:
            print('Invalid key pressed.')
        cli.print_usage()
        key = input()
