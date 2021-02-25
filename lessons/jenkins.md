## Jenkins: Pipelinefile

Below here is the file that you must add to your branch, however there are a few changes that must be made for your
branch. Create a file named `Jenkinsfile` at the root of the repository so in our case `chemistrygym/Jenkinsfile` and copy the text below into the file. Please then make the appropriate
changes highlighted by the comments. This file tells jenkins what to buildand as such you must include the file inorder 
for our tests to run.
```
pipeline {

    agent any

    stages {
        stage('Cleanup Workspace') {
            steps {
                cleanWs()
                sh """
                echo "Cleaned Up Workspace For Project"
                """
            }
        }
        stage('pull git'){
            steps {
                // !!!!!! Change branch name below to match the branch you are working in !!!!!!
                git branch: 'YOUR_BRANCH_HERE', credentialsId: '1de24870-cdba-4190-a8fb-f0508d298280', url: 'https://github.com/CLEANit/chemistrygym.git'
            }
        }
        stage('set up venv and install dependencies'){
            steps {
                sh 'python -m venv test_env'
                sh 'ls test_env'
                sh 'ls test_env/bin'
                sh '. ./test_env/bin/activate'
                sh 'pip3 install -r requirements.txt'
            }
        }
        stage(' Unit Testing Reaction') {
            steps {
                sh 'cd tests/unit/reaction_bench'
                sh 'python3 -m unittest discover -p "*_test.py"'
            }
        }
        stage(' Unit Testing Extraction') {
            steps {
                sh 'cd ../tests/unit/extraction_bench'
                sh 'python3 -m unittest discover -p "*_test.py"'
            }
        }
        // if you have another unit test directory that is not being run in the build you may use the sample code below to test it
        /*
        stage('Unit Testing TEST FILE') {
            steps {
                sh 'cd ../tests/unit/TEST_DIRECTORY'
                sh 'python3 -m unittest discover -p "*_test.py"'
            }
        }
        */

    }   
}

```