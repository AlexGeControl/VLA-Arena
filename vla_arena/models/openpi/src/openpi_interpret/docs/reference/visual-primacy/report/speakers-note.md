# Visual Primacy in VLA Policies — Speaker's Note

> Slide layout: left = InterpreT video (head 1, layer 0 → 17); right = two key findings.

---

**[Opening — gesture to the video]**

What you're seeing on the left is a live walkthrough of Pi-Zero's attention patterns using our VLA InterpreT visualization tool. I've fixed attention head 1 and I'm stepping through the backbone from the first layer all the way to the last.

Notice how at the early layers, essentially all of the attention — over 99% — is directed at the visual tokens. As we go deeper, the model gradually brings in language and state information, but even at the final layer, the majority of heads remain visually dedicated:
- Two out of eight heads at layer 17 allocate less than 1% of their attention to vision — they've become specialists for language and state. 
- But the other five still dedicate over 70% to visual tokens.

**[Transition to right side — first key point]**

This visual dominance isn't just an internal pattern — it translates directly to task performance. On the Cosmos Reason2 baseline, we ran a controlled experiment where the only change was adding a wrist camera view alongside the top-down view. Same frozen backbone, same action expert, same training. The result: an 18 percentage point improvement in success rate — from 64% to 82%. That single visual change outperformed every architectural modification to the action expert that had been tried before.

**[Second key point]**

Putting these together: the Pi-Zero analysis tells us *where* the model looks — overwhelmingly at visual tokens, validated by positive silhouette scores at every layer. The Cosmos experiment tells us that *enriching what it sees* is the most effective lever for improving performance.

**[Closing]**

Our conclusion is that the action expert's primary job is to route and decode visual information — not to build cross-modal alignment. What the model sees matters more than how it reasons about what it sees.

---

*Duration: ~90 seconds at conversational pace.*
