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
