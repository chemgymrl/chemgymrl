[chemgymrl.com](https://chemgymrl.com/)

# Troubleshooting

You may run into some unexpected errors when running chemgymrl environments from time to time. 

Here we will discuss some possible error messages here and try to find a solution to them.

## Errors

**Not adding an entrypoint when registering an environment**

```
register(
    id='WurtzExtract-v1',
    entry_point='chemistrylab.extract_bench.extract_bench_v1:WurtzExtract_v1',
)
```

```
AttributeError: module 'chemistrylab.extract_bench.extract_bench_v1:WurtzExtract_v1' has no attribute 'WurtzExtract_v1'
```

When registering any environment it's important to note that the corresponding entry point must also be added. In the 
picture above the entry point to `WurtzReact-v1` was expected to be the class `WurtzExtract_v1` in 
`reaction_bench_v1.py`. Failure to add the corresponding class will result in this error message.

**Not registering an environment**

```
>>> env = gym.make("WurtzReact-v1")
```

```
gym.error.UnregisteredEnv: No registered env with id: WurtzReact-v1
```

When adding a new environment it's required to also add a registration entry in `chemistrylab/__init__.py`. Failure to
properly register a new environment will result in this error message.
