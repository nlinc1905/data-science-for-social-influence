# data-science-for-social-influence
Code and resources for my course: Data Science for Social Influence

This repo does not contain the code for the following 2 projects.
* For the code for the hate speech detector Slack bot project, go here: 
https://github.com/nlinc1905/hate-speech-detector
* For the code for the news recommender & Bayesian A/B testing, go here: https://github.com/nlinc1905/news-recommender

# Setup & Installation

Be sure to create an environment and install the requirements before attempting to run anything.  
```commandline
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

#### Torch Geometric Installation Troubleshooting

There is an issue installing torch-scatter where it takes centuries to build the wheel.  To avoid this, 
follow the instructions on 
[Pytorch Geometric's documentation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).  
Use the version of PyTorch specified in requirements.txt (1.12.0).  Use pip.  To get your CUDA version, 
run `nvcc --version`.  You might also find 
[this StackOverflow question](https://stackoverflow.com/questions/67285115/building-wheels-for-torch-sparse-in-colab-takes-forever) helpful.

# My Previous Publications

My previous publications that are referenced in the course:
* [Data Science in Layman's Terms: Statistics](https://a.co/d/1LG8eEb)
* [Data Science in Layman's Terms: Machine Learning](https://a.co/d/gjLJVaU)
* [Data Science in Layman's Terms: Time Series Analysis](https://www.udemy.com/course/dsilt-time-series-analysis/)

# Links and References from the Lectures

Links from the lecture slides are listed below, under the name of the lecture they are from.

**Images and videos in the course are copyright free, no attribution required media from [Pexels](https://www.pexels.com/).**

### About this Course

[This person does not exist](https://thispersondoesnotexist.com/)

### Cognitive Biases Part 1

[Maslow's Hierarchy of Needs](https://en.wikipedia.org/wiki/Maslow%27s_hierarchy_of_needs#/media/File:Maslow's_Hierarchy_of_Needs2.svg)

[Criticism of Maslow's Hierarchy of Needs](https://en.wikipedia.org/wiki/Maslow%27s_hierarchy_of_needs#Criticism)

[Illusory truth effect paper](https://psycnet.apa.org/record/1978-02525-001)

[Availability heuristic - Hasher, Goldstein & Toppino, 1977](https://thedecisionlab.com/biases/availability-heuristic)

### Cognitive Biases Part 2

[Cognitive dissonance - Festinger & Carlsmith, 1959](https://psycnet.apa.org/record/1960-01158-001)

[Effort justification - Aronson & Mills, 1959](https://psycnet.apa.org/record/1960-02853-001)

[Cognitive dissonance - Brehm, 1956](https://psycnet.apa.org/record/1957-04251-001)

[Political party policy source - Cohen, 2003](https://psycnet.apa.org/record/2003-09138-003)

### Cognitive Biases Part 3

[Line length experiment & conformity - Asch, 1956](https://psycnet.apa.org/record/2011-16966-001)

[Confirmation bias - Nickerson, 1998](https://pages.ucsd.edu/~mckenzie/nickersonConfirmationBias.pdf)

[The Da Vinci Code, by Dan Brown](https://www.amazon.com/Vinci-Code-Robert-Langdon/dp/0307474275)

### Individual Behavior in Groups

[Bystander effect - Darley, J.M, & Latane, b., 1968](https://psycnet.apa.org/doi/10.1037/h0025589)

[Social loafing - Ringelmann, 1927](https://www.researchgate.net/profile/David-Kravitz-5/publication/209410111_Ringelmann_Rediscovered_The_Original_Article/links/0deec5384ffc87e9c4000000/Ringelmann-Rediscovered-The-Original-Article.pdf)

[Extreme Ownership: How U.S. Navy SEALs Lead and Win, by Jocko Willink and Leif Babin](https://www.amazon.com/Extreme-Ownership-U-S-Navy-SEALs/dp/1250067057)

### Influence

[Statistics on influencers](https://digitalmarketinginstitute.com/blog/20-influencer-marketing-statistics-that-will-surprise-you)

[YouTube video of the Vax game](https://www.youtube.com/watch?v=Sxjf4hbBv6g)

[Web Archive of the Vax game](http://web.archive.org/web/20210228194207/http://vax.herokuapp.com/game)

### Influence Decay and the Network Horizon

[Influence decay - Christakis and Fowler, 2007](https://www.nejm.org/doi/full/10.1056/NEJMsa066082)

### Information Spread in Social Networks

[Percolation](https://en.wikipedia.org/wiki/Percolation)

[Percolation critical threshold equation](https://en.wikipedia.org/wiki/Percolation)

### Phase Transitions in the Ising Model

[Boltzmann distribution](http://hyperphysics.phy-astr.gsu.edu/hbase/quantum/disfcn.html)

[Transfer matrix](https://en.wikipedia.org/wiki/Transfer-matrix_method)

[Ellis-Monaghan, 2010 "Phase Transisions in the Ising Model"](https://en.wikipedia.org/wiki/Transfer-matrix_method)

[Original solution to the 2D Ising model - Onsager, 1944](http://www.phys.ens.fr/~langlois/Onsager43.pdf)

[Torus image](https://commons.wikimedia.org/w/index.php?curid=1424661)

[Alternative way of solving the 2D Ising model](https://gandhiviswanathan.wordpress.com/2015/01/09/onsagers-solution-of-the-2-d-ising-model-the-combinatorial-method/)

[Phase transitions in 2D](https://dornsife.usc.edu/assets/sites/406/docs/505b/Ising.phase.transition.pdf)

[Proof that the 3D case is NP-hard - Barahona, 1982](https://iopscience.iop.org/article/10.1088/0305-4470/15/10/028)

[4D+ Ising models are NP-complete - Istrail, 2000](https://iopscience.iop.org/article/10.1088/0305-4470/15/10/028)

[Solutions for the 1D and 2D Ising model](https://stanford.edu/~jeffjar/statmech/lec4.html)

[Solutions for the 1D and 2D Ising model](https://www.thphys.uni-heidelberg.de/~wolschin/statsem21_3s.pdf)

[Solutions for the 1D and 2D Ising model](https://arxiv.org/pdf/1302.5843.pdf)

[Using the Ising model to model the spread of COVID19](https://arxiv.org/abs/2003.11860)

### Demo: The Rise of an Influencer

[Charles Pickering congressional record](https://www.congress.gov/member/charles-pickering/P000323)

[Paul Ryan congressional record](https://www.congress.gov/member/paul-ryan/R000570)

### Graph Representation Learning Intro

[Graph Representation Learning, by William L. Hamilton](https://www.amazon.com/Representation-Learning-Synthesis-Artificial-Intelligence/dp/1681739631)

[A Survey on Graph Representation Learning Methods, by Shima Khoshraftar and Aijun An](https://arxiv.org/abs/2204.01855)

[Deep Learning on Graphs, by Yao Ma and Jilian Tang](https://web.njit.edu/~ym329/dlg_book/)

### Graph Feature Engineering

[Node centrality measures 1](https://cambridge-intelligence.com/keylines-faqs-social-network-analysis/)

[Node centrality measures 2](https://faculty.ucr.edu/~hanneman/nettext/C10_Centrality.html)

### Spectral Properties and the Laplacian

[Spectral decomposition](https://programmathically.com/eigenvalue-decomposition/)

[Sebastian Raschka's blog post on PCA](https://sebastianraschka.com/Articles/2015_pca_in_3_steps.html#pca-and-dimensionality-reduction)

[Programmathically.com](https://programmathically.com/eigenvectors/)

### Graph Embeddings

[Word2Vec skip-gram refresher 1](https://www.guru99.com/word-embedding-word2vec.html#8)

[Word2Vec skip-gram refresher 2](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)

[Colab notebook on Word2Vec from scratch](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/word2vec.ipynb)

[Node2Vec paper](https://arxiv.org/abs/1607.00653?context=cs)

[DeepWalk paper](https://arxiv.org/abs/1403.6652)

[Cross-entropy explanation](https://users.cs.duke.edu/~cynthia/CourseNotes/CrossEnt.pdf)

### GNNs Part 1

[Graph Representation Learning, by William L. Hamilton](https://www.amazon.com/Representation-Learning-Synthesis-Artificial-Intelligence/dp/1681739631)

[Update & aggregate equations for various models](https://cmsa.fas.harvard.edu/wp-content/uploads/2020/09/hamilton_grl-1.pdf)

### GNNs Part 2

[Graph Attention Networks, by Velickovic et. al., 2017](https://arxiv.org/abs/1710.10903)

[Attention Is All You Need, Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)

[Graph Representation Learning, by William L. Hamilton](https://www.amazon.com/Representation-Learning-Synthesis-Artificial-Intelligence/dp/1681739631)

[Word order does not matter in masked language modeling - Sinha et. al., 2021](https://arxiv.org/abs/2104.06644)

[Deepsets - Ed Wagstaff, Martin Engelcke, & Fabian Fuchs](https://fabianfuchsml.github.io/permutationinvariance/)

[Modeling Relational Data with Graph Convolutional Neworks, Sclichtkrull et. al., 2017](https://arxiv.org/abs/1703.06103)

[MPNN paper - Gilmer et. al., 2017](https://arxiv.org/abs/1704.01212)

[Should Graph Neural Networks Use Features, Edges, Or Both?, by Faber, Lu, & Wattenhofer](https://arxiv.org/abs/2103.06857)

### Graph Convolutions & GCNs

[Kipf & Welling, 2017](https://arxiv.org/abs/1609.02907)

[Normalizing the adjacency matrix](https://math.stackexchange.com/questions/3035968/interpretation-of-symmetric-normalised-graph-adjacency-matrix)

[Simple Graph Convolution - Wu et. al., 2019](https://arxiv.org/abs/1902.07153)

### Graph Embeddings and GNNs for Dynamic Graphs

[A Survey on Graph Representation Learning Methods, by Shima Khoshraftar and Aijun An](https://arxiv.org/abs/2204.01855)

### Evaluating Graph Representations

[About the Weisfeiler Lehman algorithm](https://davidbieber.com/post/2019-05-10-weisfeiler-lehman-isomorphism-test/)

### Node Classification with GNNs Project Overview

[GitHub social network dataset](https://arxiv.org/abs/1909.13021)

[SNAP](https://snap.stanford.edu/data/github-social.html)

[Pytorch Geometric docs](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.GitHub)

### How to Fake Statistical Analysis

[FiveThirtyEight: Science Isn't Broken](https://fivethirtyeight.com/features/science-isnt-broken/#part1)

[Principles and Techniques of Data Science, by Sam Lau, Joey Gonzalenz, and Deb Nolan , chapter 18](https://www.samlau.me/test-textbook/ch/18/hyp_phacking.html#many-tests-for-one-phenomenon)

[How to Calculate Sample Size Needed for Power, by Jim Frost](https://statisticsbyjim.com/hypothesis-testing/sample-size-power-analysis/)

[A/B/n testing](https://www.split.io/glossary/a-b-n-testing/)

[Simpson's Paradox & the accompanying image](https://en.wikipedia.org/wiki/Simpson%27s_paradox)

[Racial discrimination in facial recognition](https://sitn.hms.harvard.edu/flash/2020/racial-discrimination-in-face-recognition-technology/)

[Student Loan Debt Holding Back Majority of Millennials from Homeownership](https://www.nar.realtor/newsroom/student-loan-debt-holding-back-majority-of-millennials-from-homeownership)

[National Association of Realtors 2022 Home Buyers and Sellers Generational Trends Report](https://cdn.nar.realtor/sites/default/files/documents/2022-home-buyers-and-sellers-generational-trends-03-23-2022.pdf)

[Black Swan theory](https://en.wikipedia.org/wiki/Black_swan_theory)

### Bayesian A/B Testing

[VWO Website](https://vwo.com)

### Fake News & Deepfakes

[What the Bible Says About Money](https://www.newsmax.com/finance/MKTNewsIntl/biblical-money-code-financial/2014/10/17/id/601403/)

[Coronavirus Bioweapon](https://greatgameindia.com/coronavirus-bioweapon/)

[Japanese PM Drinks Water from Fukushima](https://worldnewsdailyreport.com/japanese-pm-drinks-water-from-fukushima-nuclear-plant-to-reassure-world-that-radiation-is-safe-and-tasty/)

[Ghost of Kyiv](https://www.wionews.com/world/ghost-of-kyiv-killed-in-fighting-after-shooting-down-40-russian-jets-475309)

[Richard Nixon deepfake hoax](https://www.youtube.com/watch?v=2rkQn-43ixs)

[Disney Comes Out Against New York's Proposal to Curb Pornographic Deepfakes, by Gardner, 2018](https://www.hollywoodreporter.com/business/business-news/disney-new-yorks-proposal-curb-pornographic-deepfakes-1119170/)

[The Deep Fake of Dorian Gray: Who Owns the Likeness of Luke Skywalker?, By Carson Hicks](https://aublr.org/2022/04/the-deep-fake-of-dorian-gray-who-owns-the-likeness-of-luke-skywalker/)

[Marvel Made a Deal to Use Stan Lee's Likeness for the Next 20 Years, by Burlingame, 2022](https://comicbook.com/marvel/news/marvel-made-a-deal-to-use-stan-lees-likeness-for-the-next-20-years/)

[Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release)

[The Illustrated Stable Diffusion, by Jay Alammar](https://jalammar.github.io/illustrated-stable-diffusion/)

### How to Leverage Deepfakes for Social Influence

[Delaunay Triangulation with OpenCV](https://learnopencv.com/face-morph-using-opencv-cpp-python/)

[DeepFaceLab Colab notebook](https://github.com/chervonij/DFL-Colab)

[Deepfakes with First Order Model Method Colab notebook](https://colab.research.google.com/github/JaumeClave/deepfakes_first_order_model/blob/master/first_order_model_deepfakes.ipynb)

[FaceSwap](https://faceswap.dev/)

[DeepFaceLab](https://github.com/iperov/DeepFaceLab)

[Coqui](https://github.com/coqui-ai/tts)

[Real Time Voice Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)

[Kaggle Deepfake Detection Challenge](https://www.kaggle.com/competitions/deepfake-detection-challenge/overview)

[Meta Deepfake Detection](https://ai.facebook.com/blog/deepfake-detection-challenge-results-an-open-initiative-to-advance-ai/)

[DARPA Media Forensics](https://www.darpa.mil/program/media-forensics)

### Exploiting Data Visualization

[Information Dashboard Design, by Stephen Few](https://www.amazon.com/Information-Dashboard-Design-Effective-Communication/dp/0596100167)

[Mayra Magalhaes Gomes' article on Toptal](https://www.toptal.com/designers/data-visualization/data-visualization-best-practices)

[Bad visualizations](https://www.tumblr.com/badvisualisations)

[Good and bad visualizations](https://www.oldstreetsolutions.com/good-and-bad-data-visualization)

### Media Bias

[David Welch on propaganda in WWI](https://www.bl.uk/world-war-one/articles/patriotism-and-nationalism)

[Radio Act of 1927](https://www.mtsu.edu/first-amendment/article/1091/radio-act-of-1927)

[FCC fairness doctrine, 1949](https://supreme.justia.com/cases/federal/us/412/94/)

[Joe McCarthy and the Press, by Edwin Bayley](https://www.amazon.com/Joe-McCarthy-Press-Edwin-Bayley/dp/0299086240)

[The Press and Joe, by Robert Mccord](https://www.thecrimson.com/article/1982/1/11/the-press-and-joe-pbjboseph-mccarthy-americas/)

[Neil Degrasse Tyson on YouTube](https://youtu.be/S58vlJwhwDw?t=201)

[Watchdog groups](https://en.wikipedia.org/wiki/Category:Government_watchdog_groups_in_the_United_States)

### Propaganda

[Examples of historical propaganda](https://www.canva.com/learn/examples-of-propaganda/)

[History of propaganda](https://en.wikipedia.org/wiki/History_of_propaganda)

[Quotes about propaganda](https://www.physics.smu.edu/pseudo/Propaganda/)

[Examples of successful propaganda](https://bppblog.com/2018/03/23/measuring-media-impact-a-brief-history-and-analysis/)

[Bastick, 2021](https://www.sciencedirect.com/science/article/pii/S0747563220303800?via%3Dihub)

[RAND Corporation, 2020](https://www.rand.org/content/dam/rand/pubs/research_reports/RRA700/RRA704-3/RAND_RRA704-3.pdf)

[Yanez, 2020](https://bearworks.missouristate.edu/cgi/viewcontent.cgi?article=4575&context=theses)

[Martin, 1971](https://www.jstor.org/stable/1038921)

[Arceneaux & Truex, 2020](https://static1.squarespace.com/static/5431e6ebe4b07582c93c48e3/t/5e45c08d5750af6b4e5bcafe/1581629583922/varceneaux+%26+rtruex+-+Implicit+Persuasion+v+SSRN.pdf)

[Bail et. al., 2019](https://www.pnas.org/doi/10.1073/pnas.1906420116)

### Censorship

[Saussurean semiotics](https://www.cs.princeton.edu/~chazelle/courses/BIB/semio2.htm)

[1984, by George Orwell](https://a.co/d/8aMZEW1)

[Statue removed](https://news.sky.com/story/statue-of-us-president-thomas-jefferson-removed-from-new-york-city-hall-over-slavery-links-12476201)

[ACLU Definition of Censorship](https://www.aclu.org/other/what-censorship)

[The Dangers and Ethics of Social Media Censorship, by Marko Mavrovic](https://www.prindleinstitute.org/2018/09/the-dangers-and-ethics-of-social-media-censorship/)

[Censorship: Some Philosophical Issues, by Kevin McCormick](https://journals.sagepub.com/doi/pdf/10.1080/03064227708532625)

[Philosophical Issues in Censorship and Intellectual Freedom, by David Ward](https://core.ac.uk/download/pdf/4817052.pdf)

### Project Overview: Hate Speech Detector

[Project GitHub repo](https://github.com/nlinc1905/hate-speech-detector)

[Kaggle Jigsaw Unintended Bias dataset](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/101421)

[Getting started with Bolt for Python (Slack)](https://slack.dev/bolt-python/tutorial/getting-started)

### Project Overview: News Recommender

[Project GitHub repo](https://github.com/nlinc1905/news-recommender)

[Microsoft News Dataset (MIND)](https://msnews.github.io/)

### Directed Influence

[Good news first or bad news first](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6072881/)

[Prospect Theory: An Analysis of Decision Under Risk, by Kahneman & Tversky, 1979](https://www.uzh.ch/cmsssl/suz/dam/jcr:00000000-64a0-5b1c-0000-00003b7ec704/10.05-kahneman-tversky-79.pdf)

### Project Overview: Directed Influence Campaign

[Game of Thrones dialogue dataset](https://github.com/shekharkoirala/Game_of_Thrones)
