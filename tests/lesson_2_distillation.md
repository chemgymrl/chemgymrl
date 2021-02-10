## Distillation Bench: Lesson 2

### Simple experiment with the distillation bench

In this tutorial we will perform two similar experiments. In one we will turn up the temperature to the maximum 
temperature for the boiling vessel. In the other we will turn in down to the minimum. We will see the difference in both
experiments and gain more intuition into how the distillation bench works.

#### Let's try increasing the temperature to the max of the boiling vessel

You'll notice that there are 3 actions in the `demo_lesson_2d.py` file, with 2 of the 3 commented out. For this tutorial 
please just comment out the actions that we are currently not performing. 

Let's perform action 1, where we turn up the temperature slightly every step with a multiplier of 6. We can do this by 
using the code:

![code](../sample_figures/lesson_2d_image0.PNG)

When we run this code we should see that for every timestep that there is a heat change of 19999.99 joules and that the 
temperature eventually reaches 1738.0 Kelvin. We end up with a final graph that looks like this:

![graph](../sample_figures/lesson_2d_image1.PNG)

As you can see all the material is boiled off from the boiling vessel into the condensation vessel.

**Again the graphs are showing the wrong values**

#### Now let's try increasing the temperature with a higher multiplier

For this part of the tutorial you're going to want to uncomment the following code:

![code](../sample_figures/lesson_2d_image2.PNG)

Now running this code gives us an interesting result. Run it yourself to see that we will get:

![error](../sample_figures/lesson_2d_image3.PNG)

We get this error as since we increased the temperature by a higher multiplier we were able to boil off all the 
materials in the boiling vessel before letting the agent finish after 20 steps. It's important to note that choosing a
very large multiplier can result in this error. 

**Is this a error we want? Should we automatically stop the running once all the materials in boiling vessel is boiled 
off?**

#### Finally let's lower the temperature

We will now lower the temperature the absolute minimum it can be and see what happens. Let's use the following code:

![code](../sample_figures/lesson_2d_image4.PNG)

Running this results in the following graph:

![graph](../sample_figures/lesson_2d_image5.PNG)

As you will notice, nothing really changes. Of course this isn't much of a surprise since by lowering the temperature 
none of the materials in the boiling vessel will boil into the condensation vessel (in this case; if you have materials 
with lower boiling points this will change). You'll notice that the minimum temperature the boiling vessel will reach is 
297 Kelvin.

This concludes a fairly simple tutorial, to once again, get an intuition of how the distillation bench works. Hopefully 
you can see the functionality of the environment. In the next lessons we will see how to achieve a high reward by 
trying to isolate a target material, in this case, dodecane, into the condensation vessel. 