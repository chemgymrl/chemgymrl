'''
ChemistryGym Logging Module

:title: logger.py

:author: Mitchell Shahen

:history: 26-05-2020

Module to provide logging services. Supported logging formats, with examples, include:
- 'default': Adds new data logs using a simple, easily understandable line-by-line structure.

    Initialized TXT Logging File at 2000-01-01 00:00:00

    2000-01-01 00:00:00 : Sample Logging Message
    2000-01-01 00:00:01 : Sample Logging Message
    2000-01-01 00:00:02 : Sample Logging Message
    .
    .
    .

- 'json': Adds logging data in a convenient JSON structure.
          The logs are held in a list of 2-element dict objects (time and logging message).

```json
    {
        "Initialized": "2000-01-01 00:00:00"
        "Message": "Initializing JSON Logging File",
        "Logging Data": [
            {
                "time": "2000-01-01 00:00:00"
                "message": "Sample Logging Message"
            },
            {
                "time": "2000-01-01 00:00:01"
                "message": "Sample Logging Message"
            },
            {
                "time": "2000-01-01 00:00:02"
                "message": "Sample Logging Message"
            },
            .
            .
            .
        ]
    }
```

- 'xml': Adds logging data in the machine-readable, XML format.
         Each log is defined in an <item> contained in a <log> attribute.

```xml
    <data>
        <log>
            <item name="InitializeMessage">2000-01-01 00:00:00 : Initialized XML Logging</item>
        </log>
        <log>
            <item name="LogMessage1">2000-01-01 00:00:00 : Sample Logging Message</item>
        </log>
        <log>
            <item name="LogMessage2">2000-01-01 00:00:01 : Sample Logging Message</item>
        </log>
        <log>
            <item name="LogMessage3">2000-01-01 00:00:02 : Sample Logging Message</item>
        </log>
        .
        .
        .
    </data>
```

- 'yaml': Adds logging data in the machine-readable, YAML format.
          Each log is present as a string object within a dict object's list.

```yaml
    - Initialization:
        - '2000-01-01 00:00:00 : Initialized Logging File'
    - Logging Data:
        - '2000-01-01 00:00:00 : Sample Logging Message'
        - '2000-01-01 00:00:01 : Sample Logging Message'
        - '2000-01-01 00:00:02 : Sample Logging Message'
        .
        .
        .
```

To effectively use this logging module, the class must be accessible and the logs must be properly
initialized and manipulated. Firstly, a logging file is only initialized, using the `initialize`
method, after the `Logger` class has been accessed and before any logs have been added. `initialize`
prepares the defined `log_file` in accordance with the requirements of the log file's format
(`log_format`). Attempting to add logs to a log file that has not been initialized will be
unsuccessful. Additionally, initializing a file, wipes that file should it already exist, so, to
prevent any unnecessary loss of data, ensure the file you are initializing is empty or does not
exist. No parameters are specified to the initialize method, rather the `log_file` and `log_format`
parameters are passed to the class object.

A sample code snippet properly demonstrating how to use this logger is included below.

```py
# import the Logger class
from chemistrylab.logger import Logger

# access the Logger class
logging = Logger(log_file="sample", log_format="default", print_messages=False)

# initialize the log file named "sample" as a default text file
logging.initialize()

# add two logging messages
logging.add_log(message="Sample Logging Message1")
logging.add_log(message="Sample Logging Message2")
```
'''

from datetime import datetime
import json
import os
import xml.etree.ElementTree as ET
import yaml

class Logger():
    '''
    Class object to generate and contribute to the logging
    files documenting the actions taken by the Engine.
    '''

    def __init__(self, log_file="", log_format="", print_messages=False):
        '''
        Constructor class object to pass arguments to various class methods.

        Parameters
        ---------------
        `log_file` : `str` (default="")
            A string representing the name and location of the desired logging file.
        `log_format` : `str` (default="")
            A string representing how the log file is to be formatted.
        `print_messages` : `bool` (default=`False`)
            A boolean indicating if messages being logged are also printed in the terminal.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        '''

        # validate the log file and logging format parameters
        self.log_file, self.log_format = self._validate_parameters(
            log_file=log_file,
            log_format=log_format
        )

        # validate the indicator specifying if messages are
        # to be printed in the terminal as well as logged
        self.print_messages = self._validate_print(
            print_messages=print_messages
        )

    @staticmethod
    def _time():
        '''
        Method to obtain the current time as a properly formatted string.

        Parameters
        ---------------
        None

        Returns
        ---------------
        `format_time` : `str`
            A string representing the formatted current time.

        Raises
        ---------------
        None
        '''

        # obtain the current time as a datetime object
        now_time = datetime.now()

        # format the datetime object as a string
        format_time = now_time.strftime("%Y-%m-%d %H:%M:%S")

        return format_time

    @staticmethod
    def _validate_parameters(log_file="", log_format=""):
        '''
        Method to validate parameters inputted to the `Logger` class.

        Parameters
        ---------------
        `log_file` : `str` (default="")
            A string representing the name and location of the desired logging file.
        `log_format` : `str` (default="")
            A string representing how the log file is to be formatted.

        Returns
        ---------------
        `log_file` : `str`
            The validated and properly formatted name and location of the desired logging file.
        `log_format` : `str`
            The validated and properly formatted string representing how the log file is formatted.

        Raises
        ---------------
        `TypeError`:
            Raised if the inputted log file and/or logging format parameters are not strings.
        '''

        # make lists of valid formats and valid extensions
        valid_formats = ["default", "json", "xml", "yaml"]
        valid_extensions = [".txt", ".json", ".xml", ".yaml"]

        # ensures a string value for log_format when no value is provided
        log_format = log_format if log_format else ""

        # ensure that both input parameters are of the proper type
        if any([not isinstance(log_file, str), not isinstance(log_format, str)]):
            raise TypeError

        # simplify the logging format parameter
        log_format = log_format.lower()

        # split the filepath into the filename and its extension
        [filepath, extension] = os.path.splitext(log_file)

        # if no log format or file extension is found, the default setting is used
        if log_format not in valid_formats:
            if extension not in valid_extensions:
                log_format = "default"
            else:
                if extension == ".txt":
                    log_format = "default"
                elif extension == ".json":
                    log_format = "json"
                elif extension == ".xml":
                    log_format = "xml"
                elif extension == ".yaml":
                    log_format = "yaml"
                else:
                    print("The logging format, {}, is not currently supported.".format(log_format))
                    print("The default logging format will be used.")
                    log_format = "default"
                    extension = ".txt"

        # assign the proper extension to the filepath, if necessary
        if log_format == "default":
            log_file = filepath + ".txt"
        if log_format == "json":
            log_file = filepath + ".json"
        if log_format == "xml":
            log_file = filepath + ".xml"
        if log_format == "yaml":
            log_file = filepath + ".yaml"

        return log_file, log_format

    @staticmethod
    def _validate_print(print_messages):
        '''
        '''

        if not isinstance(print_messages, bool):
            raise TypeError

        return print_messages

    def _create_default(self, log_file=""):
        '''
        Method to initialize a logging file that uses the default, line-by-line, formatting.

        Parameters
        ---------------
        `log_file` : `str` (default="")
            A string representing the name and location of the desired logging file.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        '''

        # create the new dict object
        exec_time = self._time()
        formatted_message = "{} : Initialized a TXT Logging File\n".format(exec_time)

        with open(log_file, "w") as open_file:
            open_file.write(formatted_message)

    def _create_json(self, log_file):
        '''
        Method to initialize a logging file that uses JSON formatted structures.

        Parameters
        ---------------
        `log_file` : `str` (default="")
            A string representing the name and location of the desired logging file.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        '''

        exec_time = self._time()

        # create the log file and add initialization data
        initial_log = {
            "Initialized": exec_time,
            "Message": "Initializing JSON Logging File",
            "Logging Data": []
        }
        with open(log_file, 'w') as open_file:
            json.dump(initial_log, open_file, ensure_ascii=True, indent=4)

    def _create_xml(self, log_file):
        '''
        Method to initialize a logging file that uses JSON formatted structures.

        Parameters
        ---------------
        `log_file` : `str` (default="")
            A string representing the name and location of the desired logging file.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        '''

        exec_time = self._time()

        # define the overall root element to contain all logging data
        data = ET.Element("data")

        # attach a subelement to the root element
        items = ET.SubElement(data, 'log')
        item1 = ET.SubElement(items, 'item')

        # define and add an initialization logging message to the subelement
        item1.set('name', 'InitializeMessage')
        item1.text = '{} : Initialized XML Logging'.format(exec_time)

        # convert the element tree structure to bytes
        mydata = ET.tostring(data)

        # write the byte string to the log file
        with open(log_file, "wb") as open_file:
            open_file.write(mydata)

    def _create_yaml(self, log_file):
        '''
        Method to initialize a YAML file to contain logging data.

        Parameters
        ---------------
        `log_file` : `str` (default="")
            A string representing the name and location of the desired logging file.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        '''

        # get the time at execution
        exec_time = self._time()

        # define the initialization message
        initialize_message = [
            {"Initialization" : ["{} : Initialized Logging File".format(exec_time)]},
            {"Logging Data" : []}
        ]

        # open the desired yaml file and deposit the initialization message
        with open(log_file, 'w') as open_file:
            yaml.dump(initialize_message, open_file)

    def initialize(self):
        '''
        Method to determine the filename and type of formatting of the intended logging file.

        Parameters
        ---------------
        None

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        '''

        log_file = self.log_file
        log_format = self.log_format

        # ensure the file has not yet been created
        if os.path.exists(log_file):
            os.remove(log_file)

        if log_format == "default":
            self._create_default(log_file)
        if log_format == "json":
            self._create_json(log_file)
        if log_format == "xml":
            self._create_xml(log_file)
        if log_format == "yaml":
            self._create_yaml(log_file)

    def _format_default(self, message=""):
        '''
        Method to format a logging message using the default method:
        individual strings printed in a text file.

        Parameters
        ---------------
        `message` : `str` (default="")
            A string representing the logging message to be added to the logging file.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        '''

        # define the logging filename and location
        log_file = self.log_file

        # create the new dict object
        exec_time = self._time()
        formatted_message = "\n{} : {}".format(exec_time, message)

        with open(log_file, "a") as open_file:
            open_file.write(formatted_message)

    def _format_json(self, message=""):
        '''
        Method to format a logging message as a JSON structure to be placed in a .json file.

        Parameters
        ---------------
        `message` : `str` (default="")
            A string representing the logging message to be added to the logging file.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        '''

        # define the logging filename and location
        log_file = self.log_file

        # create the new log as a dict object
        exec_time = self._time()
        formatted_message = {
            "time": exec_time,
            "message": message
        }

        # open the log file with the existing logged data
        with open(log_file) as open_file:
            data = json.load(open_file)

        # append the new log as a list element in the dict of existing logs
        logging_data = data["Logging Data"]
        logging_data.append(formatted_message)
        data["Logging Data"] = logging_data

        # delete the outdated logging file
        os.remove(log_file)

        # place the updated logging file back to it's original location
        with open(log_file, "a") as open_file:
            json.dump(
                data,
                open_file,
                ensure_ascii=True,
                indent=4
            )

    def _format_xml(self, message=""):
        '''
        Method to format a piece of logging data to be added to an XML file.

        Parameters
        ---------------
        `message` : `str` (default="")
            A string representing the logging message to be added to the logging file.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        '''

        # define the log file and the current time
        log_file = self.log_file
        exec_time = self._time()

        # extract the existing log file's tree structure
        tree = ET.parse(log_file)
        root = tree.getroot()

        # define a new log element attached to the root element
        log = ET.SubElement(root, "log")

        # attach a subelement to the log element
        item = ET.SubElement(log, "item")

        # add the item name and the log message
        item.set("name", "LogMessage{}".format(len(root)))
        item.text = "{} : {}".format(exec_time, message)

        # write the modified log file to the original tree structure
        tree.write(log_file)

    def _format_yaml(self, message=""):
        '''
        Method to format a logging message and add to an existing YAML logging file.

        Parameters
        ---------------
        `message` : `str` (default="")
            A string representing the logging message to be added to the logging file.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        '''

        # define the desired logging file and the time of execution
        log_file = self.log_file
        exec_time = self._time()

        # open the existing log file, extract the logging
        # data, then close and delete the outdated file
        open_file = open(log_file)
        data = yaml.full_load(open_file)
        open_file.close()
        os.remove(log_file)

        # extract the existing logging data and append the new message to it
        logging_data = data[-1]["Logging Data"]
        logging_data.append("{} : {}".format(exec_time, message))

        # insert the updated logging data into the data object
        data[-1]["Logging Data"] = logging_data

        # recreate the yaml file and add the updated data object to it
        with open(log_file, 'w') as open_file:
            yaml.dump(data, open_file)

    def add_log(self, message=""):
        '''
        Method to add a logging message in a file as directed by the `log_file` attribute.
        The formatting of the logging data is dependent on the `log_format` attribute.

        Parameters
        ---------------
        `message` : `str` (default="")
            A string representing the logging message to be added to the logging file.

        Returns
        ---------------
        None

        Raises
        ---------------
        None
        '''

        log_file = self.log_file
        log_format = self.log_format

        if self.print_messages:
            print(message)

        if not os.path.exists(log_file):
            print(
                "\nWARNING: The logging file has not been initialized. "
                "The `initialize` attribute must be executed before "
                "attempting to add the first log to a new logging file."
            )
        else:
            if log_format == "default":
                self._format_default(message=message)

            if log_format == "json":
                self._format_json(message=message)

            if log_format == "xml":
                self._format_xml(message=message)

            if log_format == "yaml":
                self._format_yaml(message=message)
