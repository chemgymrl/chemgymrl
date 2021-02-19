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
        stage(' Unit Testing') {
            steps {
                sh 'python -V'
            }
        }
        stage(' Unit Testing_2') {
            steps {
                sh 'ls'
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
