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
                git branch: 'nicholas_fix', credentialsId: '1de24870-cdba-4190-a8fb-f0508d298280', url: 'https://github.com/CLEANit/chemistrygym.git'
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
        stage(' Unit Testing_3') {
            steps {
                sh 'pwd'
            }
        }
        stage('Code Analysis') {
            steps {
                sh """
                echo "Running Code Analysis"
                """
            }
        }

        stage('Build Deploy Code') {
            when {
                branch 'develop'
            }
            steps {
                sh """
                echo "Building Artifact"
                """

                sh """
                echo "Deploying Code"
                """
            }
        }

    }   
}
