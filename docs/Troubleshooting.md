[chemgymrl.com](https://chemgymrl.com/)

# Troubleshooting

You may run into some unexpected errors when running chemgymrl environments from time to time. 

Here we will discuss some possible error messages here and try to find a solution to them.

## Errors

**Not adding an entrypoint when registering an environment**

```
register(
    id='GenWurtzReact-v2',
    entry_point='chemistrylab.benches.reaction_bench:GeneralWurtzReact_v2'
)
```

```
AttributeError: module 'chemistrylab.extract_bench.extract_bench_v1:WurtzExtract_v1' has no attribute 'WurtzExtract_v1'
```

When registering any environment it's important to note that the corresponding entry point must also be added. In the 
picture above the entry point to `GenWurtzReact-v2` was expected to be the class `GenWurtzReact_v2` in 
`reaction_bench.py`. Failure to add the corresponding class will result in this error message.

**Not registering an environment**

```
>>> env = gym.make("GenWurtzReact-v2")
```

```
gym.error.UnregisteredEnv: No registered env with id: GenWurtzReact-v2
```

When adding a new environment it's required to also add a registration entry in `chemistrylab/__init__.py`. Failure to
properly register a new environment will result in this error message.
