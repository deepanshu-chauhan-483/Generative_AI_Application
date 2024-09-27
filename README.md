A Synopsis on

Generative AI App

(TEXT TO IMAGE GENERATION MERN STACK WEB APPLICATION)

Submitted in partial fulfillment of the requirements
for the award of the degree of Bachelor of Technology in
Computer Science and Engineering (AIML)
by

Deepanshu Chauhan 

Semester — V
Under the Supervision of

Prof. Arti Ranjan

Galgotias College of Engineering & Technology
Greater Noida, 201306

Affiliated to

Dr. APJ Abdul Kalam Technical University, Lucknow

INDEX

CONTENT PAGE NO.
ABSTRACT........cccccccnscccecscsenscececcccscacnceccanscecscectcnscecsccccnccssesenscsscecsacnsesensereseseses 1
1. INTRODUCTION. ..........:ccccccscnsorcscecencnccececncnsercscsceueuccecscncecscccncassscscecucuces 2
2. LITERATURE SURVEY. .........ccccesssnsvcccccvnsescscscscensvevcssoscccesesssssccceesnsassooees 4
3. PROBLEM FORMULATIION...........cccccncnsssccccovscssscccccccccccscccscencnsccccccsssensens 5
4. OBJECTIVES. .........cccesesercccccccscsvccccesesessscscscsssssvcccseseccvecccccnsessreccsessssecsces 6

6. ACTIVITY CHART..........ccccccccsscccccccncccncececcssesscecececsscececssseseesesaceceneesecnces 11

7. RESULTS AND DISCUSSIONG. ...........cccccccencerccncecccsccscececsccscececsenscecsecesecens 12

8. CONCLUSION ........scccccsscsccccececescsescsnencuscscccevesessseseenensescsccnssscsseensnscswewens 17

9. REFERENCEG........sccccccccscvccccccncntcscecevensnccccccnvessssscecesenscsccsesessssscscesessenes 18
ABSTRACT

This project is a MERN stack web application that utilizes OpenAI’s Dall-E model to generate AI
images based on user text prompts. The app has a modern and minimal design, with a dynamic
image layout. The user can input text prompts, and the app will generate AI images based on the
text. The user can also customize the image layout, allowing them to create unique and creative
images. The app features a gallery of AI images created by other users, allowing users to explore
and discover new images. The app also has a search function, allowing users to quickly find images
based on keywords. The tech stacks used in this project are OpenAI - Dall-E Model, Cloudinary,
MongoDB, React, Tailwind CSS, HTMLS, and Vite .



1. INTRODUCTION

The purpose of this project is to create a web application that generates AI images based on user
text prompts. The app is built using the MERN stack, which is a popular web development stack
that includes MongoDB, Express.js, React, and Node.js. The MERN stack is ideal for building web
applications that require real-time data updates and high performance. The app utilizes OpenAI’s
Dall-E model to generate images. Dall-E is a neural network that can generate images from textual
descriptions. The app has a modern and minimal design, with a dynamic image layout. The user
can input text prompts, and the app will generate AI images based on the text. The user can also
customize the image layout, allowing them to create unique and creative images.

The app features a gallery of AI images created by other users, allowing users to explore and
discover new images. The gallery is a great way to showcase the app’s capabilities and inspire users
to create their own images. The app also has a search function, allowing users to quickly find
images based on keywords. The search function is powered by MongoDB, which is a popular
NoSQL database that is ideal for storing and retrieving large amounts of data.

The app is designed to be user-friendly and easy to use. The user interface is intuitive and easy to
navigate. The app is also responsive, meaning it can be used on a variety of devices, including
desktops, laptops, tablets, and smartphones. The app is built using Tailwind CSS, which is a
utility-first CSS framework that makes it easy to create responsive and mobile-first designs.

In conclusion, this project is a great example of how the MERN stack and OpenAI’s Dall-E model
can be used to create a powerful and user-friendly web application. The app is designed to be easy
to use and features a modern and minimal design. The app’s gallery and search function make it
easy for users to explore and discover new images. The app is available on GitHub, where users
can find installation instructions, usage, contributing, tests, and license details.



Objectives


The Project has the following objectives:

= To build a web application that generates AI images based on user text prompts.

> To create a modern and minimal design with a dynamic image layout.

~ To allow users to customize the image layout, allowing them to create unique and creative images.
> To feature a gallery of AI images created by other users, allowing users to explore and discover new
images.

To have a search function, allowing users to quickly find images based on keywords.

The app is built using the MERN stack, which is a popular web development stack that includes
MongoDB, Express.js, React, and Node.js. The MERN stack is ideal for building web applications
that require real-time data updates and high performance. The app utilizes OpenAI’s Dall-E model
to generate images. Dall-E is a neural network that can generate images from textual descriptions.
The app has a modern and minimal design, with a dynamic image layout. The user can input text
prompts, and the app will generate AI images based on the text. The user can also customize the
image layout, allowing them to create unique and creative images.


The app features a gallery of AI images created by other users, allowing users to explore and discover new
images. The gallery is a great way to showcase the app’s capabilities and inspire users to create their own
images. The app also has a search function, allowing users to quickly find images based on keywords. The
search function is powered by MongoDB, which is a popular NoSQL database that is ideal for storing and
retrieving large amounts of data.

In conclusion, the objectives of the MERN stack web application that utilizes OpenAI’s Dall-E model to
generate AI images based on user text prompts are to create a modern and minimal design with a dynamic
image layout, allow users to customize the image layout, feature a gallery of AI images created by other
users, and have a search function. The app is built using the MERN stack and utilizes OpenAI’s Dall-E
model to generate images.

2. LITERATURE SURVEY

This app will be using Text-to-image generation Text-to-image generation (TTI) refers to the usage
of models that could process text input and generate high fidelity images based on text descriptions.
TTI has been an active area of research in recent years, with many models being proposed to tackle
the problem. One such model is DALL-E, which is a machine-learning model created by OpenAI
to produce images from language descriptions. DALL-E is a multimodal implementation of GPT-3
with 12 billion parameters, trained on text-image pairs from the Internet. Another model is CLIP,
which is a neural network that can generate text embeddings using contrastive learning. Contrastive
learning is a technique that learns representations by contrasting positive and negative examples. In
the case of CLIP, the positive examples are the text and image pairs, and the negative examples are
random text and image pairs. The CLIP model is trained on a large dataset of text and image pairs,
and it learns to generate text embeddings that are semantically meaningful 123. A survey of AI
text-to-image generation in the era of large models has been conducted by Fengxiang Bie et all.
The survey delves into the different types of text-to-image generation methods and provides a
detailed comparison and critique of these methods. The survey also offers possible pathways of
improvement for future work. Another survey by Chenshuang Zhang et al. reviews text-to-image
diffusion models in the context that diffusion models have emerged to be popular for a wide range
of generative tasks 2. The survey starts with a brief introduction of how a basic diffusion model
works for image synthesis, followed by how condition or guidance improves learning. Based on
that, the survey presents a review of state-of-the-art methods on text-conditioned image synthesis,
i.e., text-to-image. The survey further summarizes applications beyond text-to-image generation:
text-guided creative generation and text-guided image editing. Beyond the progress made so far, the
survey discusses existing challenges and promising future directions.
3. PROBLEM FORMULATION

The problem formulation for the project of detecting textual data from scene text images in
low-quality images is as follows:

Input: A text prompt describing the image we want to generate.

Output: Al-generated images based on the user’s text prompts.

A modern and minimal design with a dynamic image layout.

Customizable image layout, allowing users to create unique and creative images.

A gallery of AI images created by other users, allowing users to explore and discover new
images.

A search function, allowing users to quickly find images based on keywords.

Constraints:

The app should be built using the MERN stack, which includes MongoDB, Express.js, React, and
Node.js.

The app should utilize OpenAI’s Dall-E model to generate images.

The app should be designed to be user-friendly and easy to use, with a responsive user interface.
The app should be available on GitHub, where users can find installation instructions, usage,
contributing, tests, and license details.

Challenges:

Training the Dall-E model on vast amounts of text-image pair data.

Generating semantically meaningful text embeddings using CLIP.

Designing a modern and minimal user interface that is easy to navigate.

Building a dynamic image layout that can accommodate different image sizes and aspect ratios.
Implementing a search function that can quickly find images based on keywords.

Proposed Solution:

PeN>S

Using pre-trained Dall-E models to generate images.

Using CLIP to generate semantically meaningful text embeddings.

Using Tailwind CSS to design a modern and minimal user interface that is easy to navigate.

Using CSS Grid to build a dynamic image layout that can accommodate different image sizes and aspect
ratios.

Using MongoDB to store and retrieve images and implement a search function that can quickly find
images based on keywords.
Evaluation:

The problem formulation for the MERN stack web application that utilizes OpenAI’s Dall-E model to generate AI
images based on user text prompts is well-defined and comprehensive. The problem statement clearly outlines the
objectives of the app, including generating AI images based on user text prompts, creating a modern and minimal
design with a dynamic image layout, allowing users to customize the image layout, featuring a gallery of AI
images created by other users, and having a search function. The constraints of the app are also clearly defined,
including using the MERN stack, utilizing OpenAI’s Dall-E model to generate images, designing a user-friendly
interface, and making the app available on GitHub. The challenges of building such an app are also identified,
including training the Dall-E model on vast amounts of text-image pair data, generating semantically meaningful
text embeddings using CLIP, designing a modern and minimal user interface that is easy to navigate, building a
dynamic image layout that can accommodate different image sizes and aspect ratios, and implementing a search
function that can quickly find images based on keywords. The proposed solutions to these challenges are also
provided, including using pre-trained Dall-E models to generate images, using CLIP to generate semantically
meaningful text embeddings, using Tailwind CSS to design a modern and minimal user interface that is easy to
navigate, using CSS Grid to build a dynamic image layout that can accommodate different image sizes and aspect
ratios, and using MongoDB to store and retrieve images and implement a search function that can quickly find
images based on keywords

4.0OBJECTIVES

The MERN stack web application that utilizes OpenAI’s Dall-E model to generate AI images based
on user text prompts has the following objectives:

® To build a web application that generates AI images based on user text prompts.

e To create a modern and minimal design with a dynamic image layout.

© To allow users to customize the image layout, allowing them to create unique and creative
images.

e To feature a gallery of AI images created by other users, allowing users to explore and
discover new images.

e To have a search function, allowing users to quickly find images based on keywords.

The app is built using the MERN stack, which is a popular web development stack that includes
MongoDB, Express.js, React, and Node.js. The MERN stack is ideal for building web applications
that require real-time data updates and high performance. The app utilizes OpenAI’s Dall-E model
to generate images. Dall-E is a neural network that can generate images from textual descriptions.
The app has a modern and minimal design, with a dynamic image layout. The user can input text
prompts, and the app will generate AI images based on the text. The user can also customize the
image layout, allowing them to create unique and creative images.

The app features a gallery of AI images created by other users, allowing users to explore and
discover new images. The gallery is a great way to showcase the app’s capabilities and inspire users
to create their own images. The app also has a search function, allowing users to quickly find
images based on keywords. The search function is powered by MongoDB, which is a popular
NoSQL database that is ideal for storing and retrieving large amounts of data.

In conclusion, the objectives of the MERN stack web application that utilizes OpenAI’s Dall-E
model to generate AI images based on user text prompts are to create a modern and minimal design
with a dynamic image layout, allow users to customize the image layout, feature a gallery of AI
images created by other users, and have a search function. The app is built using the MERN stack
and utilizes OpenAI’s Dall-E model to generate images.

4. METHODOLOGY

e The app is built using the MERN stack, which is a popular web development stack that
includes MongoDB, Express.js, React, and Node.js. MongoDB is used as the database, and
React is used for the front-end. Tailwind CSS is used for styling, and Vite is used for
building and serving the app.

e@ The app utilizes OpenAI’s Dall-E model to generate images. Dall-E is a neural network that
can generate images from textual descriptions. The input to the Transformer model is a
sequence of tokenized image caption followed by tokenized image patches. The model is
trained on vast amounts of text-image pair data and uses an optimization process to
fine-tune its parameters. This optimization process is essentially a feedback loop where the
model predicts an output, compares it to the actual output, calculates the error, and adjusts
the model parameters to minimize this error 12.

e Cloudinary is used to store and manage the images. Cloudinary is a cloud-based image and
video management service that provides a comprehensive set of tools for managing media
assets 3.

e The app is designed to be user-friendly and easy to use. The user interface is intuitive and
easy to navigate. The app is also responsive, meaning it can be used on a variety of devices,
including desktops, laptops, tablets, and smartphones.

e@ The app features a gallery of AI images created by other users, allowing users to explore
and discover new images. The gallery is a great way to showcase the app’s capabilities and
inspire users to create their own images. The app also has a search function, allowing users
to quickly find images based on keywords. The search function is powered by MongoDB,
which is a popular NoSQL database that is ideal for storing and retrieving large amounts of
data.

Architecture Overview

The architecture, at a higher level, is quite easy to understand and consists of 3 parts
Text Encoder : To encode text data into text embeddings

Prior: Using text embeddings , generate image embeddings (the bridge)

Decoder: Generating image using image embeddings
TEXT-IMAGE pair

CLIP

Text Encoder Image Encoder
let} 9

|__y

TEXT + IMAGE
embedding <

Vv

Diffusion Prior ,
Transformer Decoder

DECODER , GLIDE

The training process, the text-image pair represents the training dataset

TEXT
Prompt

> CLIP's Text Encoder

Text Embedding |

TEXT + Random IMAGE embedding

¥

PRIOR, Decoder only
Transformer

Text + Image embedding

¥v

DECODER, GLIDE

Final DALL-E architecture. Observe Image Encoder has been removed

Ad

IMAGE

How DALL-E 2 Works: A Detailed Look

Now it's time to dive into each of the above steps separately. Let's get started by looking at how
DALL-E 2 learns to link related textual and visual abstractions.

Step 1 - Linking Textual and Visual Semantics
After inputting "a teddy bear riding a skateboard in Times Square", DALL-E 2 outputs the
following image:

How does DALL-E 2 know how a textual concept
like "teddy bear" is manifested in the visual space?
The link between textual semantics and their visual
representations in DALL-E 2 is learned by another
OpenAI model called CLIP (Contrastive
Language-Image Pre-training).

CLIP is trained on hundreds of millions of images
and their associated captions, learning how much a
given text snippet relates to an image. That is, rather
than trying to predict a caption given an image, CLIP
instead just learns how related any given caption is to
an image. This contrastive rather than predictive
objective allows CLIP to learn the link between
textual and visual representations of the same abstract object. The entire DALL-E 2 model hinges
on CLIP's ability to learn semantics from natural language, so let's take a look at how CLIP is
trained to understand its inner workings.

CLIP Training

The fundamental principles of training CLIP are quite simple:

1. First, all images and their associated captions are passed through their respective
encoders, mapping all objects into an m-dimensional space.

2. Then, the cosine similarity of each (image, text) pair is computed.

3. The training objective is to simultaneously maximize the cosine similarity between N
correct encoded image/caption pairs and minimize the cosine similarity between N2 - N
incorrect encoded image/caption pairs.

This training process is visualized below:

Overview of the CLIP training process

Additional Training Details

Significance of CLIP to DALL-E 2

CLIP is important to DALL-E 2 because it is what ultimately determines how semantically-related
a natural language snippet is to a visual concept, which is critical for text-conditional image
generation.

Additional Information
Step 2 - Generating Images from Visual Semantics

After training, the CLIP model is frozen and DALL-E 2 moves onto its next task - learning to
reverse the image encoding mapping that CLIP just learned. CLIP learns a representation space in
which it is easy to determine the relatedness of textual and visual encodings, but our interest is in
image generation. We must therefore learn how to exploit the representation space to accomplish
this task.

In particular, OpenAI employs a modified version of another one of its previous models, GLIDE, to
perform this image generation. The GLIDE model learns to invert the image encoding process in
order to stochastically decode CLIP image embeddings.

encoder

An
image of a Corgi playing a flamethrowing trumpet passed through CLIP's image encoder. GLIDE
then uses this encoding to generate a new image that maintains the salient features of the original.
(modified from source)

As depicted in the image above, it should be noted that the goal is not to build an autoencoder and
exactly reconstruct an image given its embedding, but to instead generate an image which
maintains the salient features of the original image given its embedding. In order perform this
image generation, GLIDE uses a Diffusion Model.

What is a Diffusion Model?

Diffusion Models are a thermodynamics-inspired invention that have significantly grown in
popularity in recent years[1][2]. Diffusion Models learn to generate data by reversing a gradual
noising process. Depicted in the figure below, the noising process is viewed as a parameterized
Markov chain that gradually adds noise to an image to corrupt it, eventually (asymptotically)
resulting in pure Gaussian noise. The Diffusion Model learns to navigate backwards along this
chain, gradually removing the noise over a series of timesteps to reverse this process.

Po(X+-1|Xt)
Om -&)-—-@-—- a

aoalen, 1)

Diffusion Model schematic (source).
If the Diffusion Model is then "cut in half" after training, it can be used to generate an image by
randomly sampling Gaussian noise and then de-noising it to generate a photorealistic image. Some
may recognize that this technique is highly reminiscent of generating data with Autoencoders, and
Diffusion Models and Autoencoders are, in fact, related.

GLIDE Training

While GLIDE was not the first Diffusion Model, its important contribution was in modifying them
to allow for text-conditional image generation. In particular, one will notice that Diffusion Models
start from randomly sampled Gaussian noise. It at first unclear how to tailor this process to
generate specific images. If a Diffusion Model is trained on a human face dataset, it will reliably
generate photorealistic images of human faces; but what if someone wants to generate a face with a
specific feature, like brown eyes or blonde hair?

GLIDE extends the core concept of Diffusion Models by augmenting the training process with
additional textual information, ultimately resulting in text-conditional image generation. Let's take
a look at the training process for GLIDE:

GLIDE training process.

Additional Training Details

Here are some examples of images generated with GLIDE. The authors note that GLIDE performs
better than DALL-E (1) for photorealism and caption similarity.

“a hedgehog using a “a corgi wearing a red bowtie “robots meditating in a “a fall landscape with a small
calculator” and a purple party hat” vipassana retreat” cottage next to a lake”

Examples of images generated by GLIDE (source).

DALL-E 2 uses a modified GLIDE model that incorporates projected CLIP text embeddings in two
ways. The first way is by adding the CLIP text embeddings to GLIDE's existing timestep
embedding, and the second way is by creating four extra tokens of context, which are concatenated
to the output sequence of the GLIDE text encoder.

Significance of GLIDE to DALL-E 2

GLIDE is important to DALL-E 2 because it allowed the authors to easily port over GLIDE's
text-conditional photorealistic image generation capabilities to DALL-E 2 by instead conditioning
on image encodings in the representation space. Therefore, DALL-E 2's modified GLIDE learns to
generate semantically consistent images conditioned on CLIP image encodings. It is also important
to note that the reverse-Diffusion process is stochastic, and therefore variations can easily be
generated by inputting the same image encoding vectors through the modified GLIDE model
multiple times.
Step 3 - Mapping from Textual Semantics to Corresponding Visual Semantics

While the modified-GLIDE model successfully generates images that reflect the semantics
captured by image encodings, how do we go about actually go about finding these encoded
representations? In other words, how do we go about injecting the text conditioning information
from our prompt into the image generation process?

Recall that, in addition to our image encoder, CLIP also learns a text encoder. DALL-E 2 uses
another model, which the authors call the prior, in order to map from the text encodings of image
captions to the image encodings of their corresponding images. The DALL-E 2 authors experiment
with both Autoregressive Models and Diffusion Models for the prior, but ultimately find that they
yield comparable performance. Given that the Diffusion Model is much more computationally
efficient, it is selected as the prior for DALL-E 2.

o—~, —

( )

4 = |
< fea

>()> —>-

=

O

ay,

OOO

fH

diffusion
prior
text image

encoding encoding
Prior

mapping from a text encoding to its corresponding image encoding (modified from source).

Prior Training

The Diffusion Prior in DALL-E 2 consists of a decoder-only Transformer. It operates, with a causal
attention mask, on an ordered sequence of

The tokenized text/caption.

The CLIP text encodings of these tokens.

An encoding for the diffusion timestep.

The noised image passed through the CLIP image encoder.

Final encoding whose output from Transformer is used to predict the unnoised CLIP
image encoding.

Additional Training Details

wPwWN PD
Step 4 - Putting It All Together
At this point, we have all of DALL-E 2's functional components and need only to chain them
together for text-conditional image generation:

1. First the CLIP text encoder maps the image description into the representation space.

2. Then the diffusion prior maps from the CLIP text encoding to a corresponding CLIP
image encoding.

3. Finally, the modified-GLIDE generation model maps from the representation space into
the image space via reverse-Diffusion, generating one of many possible images that
conveys the semantic information within the input caption.

“a Corgi
playing a
QO O
flame O+O+¢
throwing OO
trumpet”

High-level overview of the DALL-E 2 image-generation process (modified from source).

Summary

DALL-E 2 exemplifies the effectiveness of Diffusion Models in the realm of Deep Learning. Both
the prior and image generation components of DALL-E 2 are based on Diffusion Models, which
have gained prominence in recent years and are poised to play a more significant role in the future
of Deep Learning research.Another key takeaway is the importance and potency of harnessing
natural language for training State-of-the-Art Deep Learning models. Although this concept is not
exclusive to DALL-E 2 (as demonstrated previously by CLIP), it's crucial to recognize that
DALL-E 2's power ultimately derives from the vast volume of paired natural language and image
data available on the internet. Utilizing such data not only eliminates the time-consuming task of
manual dataset labeling but also captures the noisy and unfiltered nature of real-world data,
enhancing the robustness of Deep Learning models.Lastly, DALL-E 2 reaffirms the dominance of
Transformers for models trained on massive web-scale datasets due to their impressive
parallelization capabilities.




5. RESULTS AND DISCUSSIONS

The challenges in this project include training the DALL-E model on vast amounts of text-image pair
data, generating semantically meaningful text embeddings using CLIP, designing a modern and minimal
user interface that is easy to navigate, building a dynamic image layout that can accommodate different
image sizes and aspect ratios, and implementing a search function that can quickly find images based on
keywords.

To address these challenges, we propose using pre-trained DALL-E models to generate images,
CLIP to generate semantically meaningful text embeddings, Tailwind CSS to design a modern and
minimal user interface that is easy to navigate, CSS Grid to build a dynamic image layout that can
accommodate different image sizes and aspect ratios, and MongoDB to store and retrieve images
and implement a search function that can quickly find images based on keywords.



6. CONCLUSION

In conclusion, the project aims to provide a user-friendly and efficient way to generate
AlI-generated images based on user’s text prompts. The proposed solution involves using
pre-trained DALL-E models, CLIP, Tailwind CSS, CSS Grid, and MongoDB to address the
challenges in this project. The project is available on GitHub, where users can find
installation instructions, usage, contributing, tests, and license details.

