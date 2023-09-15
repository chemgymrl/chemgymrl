# Testing Environment

Our testing suite is a series of test cases that are used to validate several facets of the ChemGymRL environment including the testing of benches, functions, classes and the lab manager. All these unit tests are currently accessible in the ‘chemgymrl/tests/unit’ directory and can be run independently or all together. To test all available unit tests at once make sure to run the following command:

```
cd tests/unit && python -m unittest discover -p "*test*".py
```

For more information, see the [ChemGymRL Documentation](https://docs.chemgymrl.com).
