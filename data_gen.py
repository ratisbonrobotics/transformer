import os
import time
import random
import pickle
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

curriculum = [
    "Autophagy in Cellular Health: Exploring the critical role of autophagy in cell maintenance, defense mechanisms, and implications in disease.",
    "Financial Derivatives: Understanding derivative contracts, their valuation, and strategic use in hedging risk and speculative investing.",
    "Advanced Programming Concepts: Dive into sophisticated programming techniques, algorithms, and software design principles across languages.",
    "Modern Pedagogical Techniques: Investigating contemporary teaching methods, educational technologies, and effective classroom management strategies.",
    "Stoicism and Modern Life: An exploration of Stoic philosophy, focusing on self-control, resilience, and applying Stoic principles to contemporary challenges.",
    "Photosynthesis and Plant Biology: Examining the mechanisms of photosynthesis and its impact on plant life and global ecosystems.",
    "Quantum Mechanics Fundamentals: Unraveling the principles of quantum mechanics, its mathematical frameworks, and applications in modern physics.",
    "Universal Basic Income: Prospects and Challenges: Analyzing the concept of universal basic income, its potential impacts on society and the economy.",
    "Artificial Intelligence and Society: Exploring the development of AI, its applications in various fields, and ethical considerations.",
    "Climate Change: Causes and Mitigation: Understanding the science of climate change, its effects on the planet, and strategies for mitigation and adaptation.",
    "Gene Editing Technologies: Investigating the tools and techniques of gene editing, ethical concerns, and potential applications in medicine and agriculture.",
    "Socratic Method in Critical Thinking: Developing critical thinking and reasoning skills through the Socratic method of questioning and dialogue.",
    "Einstein's Theory of Relativity: An exploration of special and general relativity, its revolutionary impact on our understanding of the universe.",
    "Sustainable Development Goals: Examining the United Nations' SDGs, their targets, and the global efforts to achieve sustainable development.",
    "Nanotechnology and Innovation: Delving into the manipulation of matter at the nanoscale and its applications in various industries, from medicine to materials science.",
    "Yoga: Theory and Practice: Studying the ancient art of Yoga, its philosophical underpinnings, and health benefits, including practical sessions.",
    "Foundations of Democracy: A comprehensive study of democratic governance, its principles, challenges, and the role of citizens.",
    "Existentialism in Literature and Philosophy: Exploring existentialist themes in philosophy and literature, focusing on human freedom, choice, and meaning.",
    "Classical Literature: A survey of seminal works from ancient civilizations, their historical contexts, and enduring influence on modern literature.",
    "World History: A Global Perspective: Tracing human civilization from ancient times to the present across different regions and cultures.",
    "Comparative Religion: An analysis of the world's major religions, their beliefs, rituals, and impacts on culture and society.",
    "Philosophy of Mind: Investigating the nature of consciousness, the mind-body problem, and debates in cognitive science and psychology.",
    "Political Theory: Key Concepts and Debates: Examining foundational ideas, ideologies, and debates that have shaped political thought.",
    "Bioethics: Dilemmas in Modern Science and Medicine: Exploring ethical issues in contemporary bioscience and healthcare practices.",
    "Introduction to Astronomy: Observing the Cosmos: A journey through the universe, studying celestial phenomena, and the tools astronomers use.",
    "Classical Music Appreciation: Exploring the evolution of Western classical music, its composers, and musical periods.",
    "Language Acquisition Theories: Understanding how humans learn languages, including cognitive, sociocultural, and biological perspectives.",
    "Sociology of Modern Societies: Analyzing the structure and dynamics of societies, social institutions, and cultural practices.",
    "Principles of Ecology: Examining ecosystems, biodiversity, and the interdependence of living organisms and their environment.",
    "Medieval History: Chronicles of the Middle Ages: Exploring the political, cultural, and social aspects of Medieval Europe and beyond.",
    "Ethnographic Field Methods: Learning ethnographic research techniques for in-depth study of cultures and communities.",
    "Classical Mechanics: Analyzing motion, forces, and energy in physical systems using mathematical models.",
    "Microeconomic Theory: Examining the decision-making processes of individuals and firms and their impact on market outcomes.",
    "Linguistics: The Science of Language: Exploring the structure, usage, and psychology of language across cultures.",
    "Archaeology: Unearthing Human Past: Introduction to archaeological methods and significant discoveries that have shaped our understanding of history.",
    "Genetics: From Genes to Genomes: Understanding the principles of heredity, genetic variations, and the structure and function of genomes.",
    "Geography: Exploring Human and Physical Landscapes: Studying the Earth's environments, landscapes, and human-geographical interactions.",
    "Art History: A Journey Through Time: Surveying visual arts from ancient times to the contemporary era, analyzing artistic movements and works.",
    "Thermodynamics: Energy, Work, and Heat: Understanding the laws of thermodynamics and their applications in various physical and chemical processes.",
    "Civil Engineering Foundations: Designing and constructing infrastructure projects, including materials science, structural analysis, and project management.",
    "Financial Markets and Instruments: An overview of financial markets, investment vehicles, and strategies for risk and return optimization.",
    "Public Health Essentials: Exploring the science of preventing disease, extending life, and promoting health through organized societal efforts.",
    "Particle Physics: The Building Blocks of Matter: Studying elementary particles and the forces governing them, uncovering the fundamental structure of the universe.",
    "World Mythologies: Understanding myths across cultures, their symbolism, and their influence on society and literature.",
    "Botany: Plant Science and Ecology: Exploring plant biology, physiology, taxonomy, and the role of plants in ecosystems.",
    "Zoology: Animal Biology and Behavior: Studying animal life, from cellular biology to ecosystems, including anatomy, physiology, evolution, and conservation.",
    "Material Science: Innovating with Materials: Investigating the properties of materials and their applications in technology and manufacturing.",
    "Theatre Arts: Performance and Production: Examining the theatrical creation process, including acting, directing, design, and playwriting.",
    "Entrepreneurship: Launching Innovative Businesses: Guiding the development, launch, and growth of new business ventures with a focus on innovation and market impact.",
    "Classical Languages and Texts: Learning ancient Greek and Latin, focusing on their literature, culture, and influence on Western civilization.",
    "Moral Philosophy: Ethics and Human Action: Investigating theories of ethics, moral reasoning, and the philosophical basis of moral judgments.",
    "Cultural Anthropology: Societies and Cultures Explained: Studying human cultures, practices, beliefs, and social structures through ethnographic research.",
    "Quantum Physics: Unveiling the Quantum World: Delving deeper into the mysteries of quantum behavior, superposition, entanglement, and quantum technologies.",
    "Veterinary Science: Animal Health and Medicine: Focusing on the diagnosis, treatment, and prevention of diseases in animals.",
    "Urban Planning: Shaping Sustainable Cities: Analyzing theories and methods for planning urban spaces, infrastructure, and policies for sustainable development.",
    "Horticulture: Cultivation and Management: Exploring the art and science of growing fruits, vegetables, flowers, and ornamental plants.",
    "Psychoanalysis and the Unconscious: Exploring Freudian theory, psychoanalytic techniques, and their influence on psychology and culture.",
    "Metaphysics: Questions of Reality and Existence: Investigating fundamental concepts such as being, existence, time, and reality.",
    "Molecular Biology: Exploring the Molecular Basis of Life: Studying the structure and function of biomolecules and the mechanisms of genetic information flow.",
    "Oceanography: Sea Exploration and Study: Examining the physical, chemical, geological, and biological processes of the ocean.",
    "Forensic Science: Crime Scene to Courtroom: Applying scientific methods to solve crimes, focusing on evidence collection, analysis, and legal implications.",
    "Nutrition Science: Diet and Health: Understanding the role of nutrients in health, disease prevention, and the relationship between diet, fitness, and well-being.",
    "Civil Rights Law: Protecting Individual Freedoms: Examining the legal protections against discrimination and the rights guaranteed by the Constitution.",
    "Information Theory: Data, Signals, and Coding: Exploring the quantification, storage, and communication of information, with applications in coding theory and telecommunications.",
    "Classical Architecture: Principles and Evolution: Studying the architecture of ancient Greece and Rome, its elements, aesthetics, and historical significance.",
    "Epistemology: Theories of Knowledge: Investigating the nature, sources, and limits of knowledge, including belief justification and truth theories.",
    "Constitutional Law: Governance and Rights: Analyzing the principles, structures, and functions of constitutional law, focusing on rights, freedoms, and government powers.",
    "Environmental Science: Ecosystems and Human Impact: Studying the interaction between humans and the environment, addressing conservation, pollution, and sustainability challenges.",
    "International Relations: Global Politics and Diplomacy: Examining the theories and practices of international politics, conflicts, cooperation, and global governance.",
    "Optics: Light and Its Applications: Exploring the properties of light, optical instruments, and applications in technology, communication, and medicine.",
    "Mechanical Engineering Essentials: From Design to Implementation: Covering the fundamentals of mechanical design, materials science, thermodynamics, and fluid mechanics.",
    "Cognitive Psychology: The Mind and Its Processes: Exploring mental processes such as perception, memory, thought, and problem-solving, with insights from neuroscience.",
    "Classical Mythology: Tales of Gods and Heroes: Analyzing myths of ancient Greece and Rome, their meanings, cultural context, and influence on Western literature.",
    "Hydrology and Water Resources: Studying the distribution, movement, and quality of water on Earth, including water resource management and conservation.",
    "Comparative Literature: Across Cultures and Epochs: Exploring literary works across different languages, cultures, and historical periods, focusing on themes and narrative techniques.",
    "Evolutionary Biology: From Genes to Species: Understanding the mechanisms of evolution, natural selection, and speciation, and their impact on biodiversity.",
    "Cybersecurity: Defending the Digital Frontier: Examining threats to information security, cybersecurity strategies, and the ethical and legal considerations in cyber defense.",
    "Molecular Genetics: Heredity at the Molecular Level: Studying the structure, regulation, and expression of genes, and their roles in inheritance and disease.",
    "Philosophy of Science: Examining the foundations, methods, and implications of science, questioning scientific reasoning and the nature of scientific knowledge.",
    "Immunology: The Body's Defense System: Understanding the immune system's components, how it fights diseases, and implications for vaccine development.",
    "Developmental Psychology: Human Growth and Development: Investigating the psychological changes that occur from infancy through adulthood and their effects on behavior.",
    "Electrical Engineering: Circuits and Systems: Covering the principles of electrical circuits, electronics, electromagnetism, and signal processing.",
    "Operations Research: Decision-Making Models: Applying mathematical and statistical methods to solve complex decision-making problems in business, engineering, and public policy.",
    "Neuroscience: Exploring the Brain and Nervous System: Understanding the structure, function, and disorders of the nervous system, with insights into behavior and cognition.",
    "Aesthetics: Philosophy of Art and Beauty: Studying the nature of art, beauty, and taste, exploring how art evokes emotions and communicates ideas.",
    "Computational Science: Modeling and Simulation: Using advanced computing to solve large-scale scientific and engineering problems, including modeling, simulation, and data analysis techniques.",
    "Paleontology: Uncovering the Fossil Record: Studying fossils to understand the history of life on Earth, evolution, and ancient environments.",
    "Urban Sociology: The City and Social Change: Analyzing urbanization, social stratification, and the impact of urban environments on social behavior and relationships.",
    "Rhetoric: The Art of Persuasion: Examining classical and modern theories of rhetoric, focusing on effective communication, argumentation, and public speaking.",
    "Cosmology: Unraveling the Universe's Mysteries: Exploring the origins, structure, evolution, and eventual fate of the universe at a cosmic scale.",
    "Behavioral Economics: Psychology Behind Economic Decisions: Investigating how psychological, cognitive, emotional, and cultural factors affect economic decision-making and market outcomes.",
    "Quantum Computing: Revolutionizing Computation: Introduction to the principles of quantum computing, quantum algorithms, and their potential to solve complex problems.",
    "Biophysics: At the Interface of Biology and Physics: Exploring the application of physical principles to biological systems, including molecular structure, dynamics, and interactions.",
    "Electromagnetism: From Maxwell's Equations to Applications: Understanding electric and magnetic fields, their interactions, and applications in technology and scientific research.",
    "Public Administration: Governing and Public Services: Examining the organization, policies, and practices of public administration in government and non-profit sectors.",
    "Solid State Physics: The Science of Materials: Investigating the physical properties of solids, including crystals, semiconductors, and superconductors.",
    "Philosophy of Law: Justice, Legality, and Morality: Analyzing the nature of law, legal reasoning, and the relationship between law, morality, and society.",
    "Patristics: The Church Fathers and their Legacy: Studying the writings and theological contributions of early Christian writers and their impact on Christian doctrine.",
    "Aeronautical Engineering: Principles of Flight and Aircraft Design: Covering the fundamentals of aerodynamics, propulsion, avionics, and materials for aircraft and spacecraft.",
    "Political Economy: Markets, States, and Societies: Exploring the interactions between political and economic systems, public policy, and economic behavior.",
    "Seismology: Earthquakes and the Earth's Interior: Investigating the causes and effects of earthquakes, seismic waves, and the structure of the Earth's interior.",
    "Bioinformatics: From Genes to Systems: Applying computational tools and techniques to understand biological data, including genomics, proteomics, and bioinformatics databases.",
    "Renaissance Literature: Rebirth of Classical Ideals: Examining the literary works and cultural movements of the Renaissance era, highlighting humanism and innovation.",
    "Philosophy of Education: Theories and Approaches: Investigating the purpose, methods, and philosophical foundations of education across different educational theories.",
    "Structural Engineering: Designing Against Failure: Study of the analysis and design of structures that must withstand loads and environmental elements.",
    "Virology: Viruses and Their Impact on Life: Exploring the structure, function, and properties of viruses and their roles in health, disease, and ecology.",
    "Biochemical Engineering: Bioprocesses and Technologies: Applying principles of chemical engineering to biological materials, focusing on bioprocessing techniques and bio-based product development.",
    "Islamic Studies: Faith, Culture, and Society: Examining Islamic beliefs, practices, history, and impact on art, science, and global societies.",
    "Robotics: Design and Control: Exploring the design, construction, operation, and application of robots, including autonomous and industrial robots.",
    "Medieval Philosophy: Thought and Legacy: Investigating the philosophical ideas and contributions of medieval thinkers, bridging ancient philosophy and the Renaissance.",
    "Plasma Physics: State of Matter and Fusion Energy: Studying plasma, the fourth state of matter, its properties, and applications, including fusion energy research.",
    "Environmental Ethics: Philosophy for a Threatened Planet: Exploring ethical considerations regarding human interactions with the natural environment, conservation, and sustainability.",
    "Quantitative Methods in Social Sciences: Applying statistical techniques and data analysis methods to understand patterns and phenomena in the social sciences.",
    "Roman Law: Foundations of Western Legal Systems: Exploring ancient Roman law, its principles, development, and lasting influence on modern legal systems.",
    "Classical Studies: Exploring the Ancient World: Investigating the languages, literature, history, and culture of ancient Greece and Rome.",
    "Philosophy of Religion: Faith, Reason, and Belief: Examining philosophical questions related to religion, including the existence of deity, religious experience, and the relationship between religion and ethics.",
    "Wildlife Conservation: Protecting Biodiversity: Studying the principles of wildlife conservation, including habitat protection, species preservation, and conservation policies.",
    "Marine Biology: Life in the Oceans: Examining the diversity, ecology, and behavior of marine organisms and their environments.",
    "Human Geography: People, Places, and Cultures: Analyzing spatial patterns in human activity and the cultural, political, and economic processes that shape landscapes.",
    "Ancient History: Civilizations and Empires: Surveying the development of ancient civilizations around the world, their impacts, and legacies.",
    "Peace Studies: Theories and Practices: Exploring approaches to peace, conflict resolution, and non-violent struggle, with case studies on conflict prevention and peacebuilding.",
    "African Studies: Continent, Cultures, and Challenges: A multidisciplinary examination of Africa, covering its history, politics, socioeconomic issues, and cultural diversity.",
    "Digital Humanities: Technology and the Arts: Integrating computing and digital technologies with humanities research, examining digital culture, and analyzing textual and visual data.",
    "East Asian Studies: Tradition and Transformation: Investigating the history, culture, politics, and societal changes in East Asian countries, with emphasis on China, Japan, and Korea.",
    "Medicinal Chemistry: From Drugs to Therapies: Exploring the design, synthesis, and mechanism of action of pharmaceutical compounds.",
    "South Asian Studies: History and Modern Dynamics: Studying the diverse cultures, religions, history, and contemporary issues of South Asian countries.",
    "Latin American Studies: Identity, Inequality, and Innovation: An interdisciplinary analysis of Latin American societies, focusing on their historical contexts, cultural expressions, and contemporary challenges.",
    "Quantitative Finance: Financial Modeling and Risk Management: Applying mathematical and statistical methods to financial market analysis, investment strategies, and risk management.",
    "Holocaust Studies: History and Memory: Examining the history of the Holocaust, its causes, dynamics, and legacies, with a focus on ethical and educational dimensions.",
    "Children's Literature: Narrative, Culture, and Impact: Analyzing children's books and media, exploring themes, pedagogical theories, and the role of children's literature in societal norms.",
    "Paleoanthropology: Human Evolution Unveiled: Studying fossil records and archaeological evidence to trace the evolutionary history of humans.",
    "Classical Philology: Texts and Interpretation: The study of ancient Greek and Latin texts, focusing on their language, literature, and historical context.",
    "Homiletics: The Craft of Preaching: Exploring the art and theology of preaching, including sermon preparation, delivery methods, and listener engagement.",
    "Comparative Mythology: Symbols and Archetypes: Analyzing myths from diverse cultures to understand universal themes, symbols, and archetypes in human storytelling.",
    "Embryology: Development from Conception to Birth: Examining the development of embryos and fetuses, including genetic and environmental influences.",
    "Philosophy of Art: Aesthetics and Critique: Investigating theories of beauty, art creation, and interpretation, and their relevance to contemporary art and culture.",
    "Patent Law and Innovation: Understanding the legal framework for protecting inventions, including the patenting process, rights, and the impact on technological innovation.",
    "Gemology: The Science of Precious Stones: Studying the properties, identification, and valuation of gemstones, including diamonds, rubies, sapphires, and emeralds.",
    "Epigraphy: Deciphering Ancient Inscriptions: Exploring the study of ancient inscriptions engraved on materials, and their historical, linguistic, and cultural significance.",
    "Hermeneutics: The Art of Interpretation: Investigating the theories and methodologies of interpreting texts, symbols, and cultural artifacts.",
    "Comparative Politics: Systems, Processes, and Power: Examining political institutions, behavior, and policies across different countries to understand comparative governance and power dynamics.",
    "Systematic Theology: Doctrine, Tradition, and Practice: Delving into Christian doctrines, exploring theological systems, and examining their historical development and practical applications.",
    "Paleography: Reading Ancient Manuscripts: Learning the skills to decipher and study historical handwriting, focusing on manuscripts from various periods.",
    "Numerical Analysis: Mathematical Algorithms and Computations: Applying numerical methods to solve mathematical problems, emphasizing algorithms and computational techniques.",
    "Iconography: Symbols and Meanings in Art: Exploring the symbolism in visual arts, interpreting icons, symbols, and narratives within cultural and historical contexts.",
    "Talmudic Studies: Texts and Interpretations: Engaging with the Talmud through study of its texts, history, and interpretations within Jewish legal tradition and thought.",
    "Diaspora Studies: Identity, Migration, and Culture: Investigating the historical and contemporary experiences of diasporic communities, focusing on identity, migration, and cultural exchange.",
    "Hebrew Bible Studies: Texts, Contexts, and Traditions: Analyzing the texts of the Hebrew Bible (Tanakh), exploring its literary, historical, and theological dimensions.",
    "Classical Rhetoric: Persuasion and Expression: Examining the principles of classical rhetoric, its historical development, and application in modern contexts.",
    "Applied Physics: Bridging Science and Technology: Exploring applications of physics in technology, engineering, and industry, focusing on practical problem-solving.",
    "Forensic Anthropology: Solving Mysteries from Bones: Applying anthropological methods to solve legal cases, identifying human remains, and understanding forensic contexts.",
    "Health Economics: Systems, Policies, and Outcomes: Analyzing the economic aspects of health care systems, including funding models, health behaviors, and policy implications.",
    "History of Philosophy: Exploring Philosophical Thought: Tracing the evolution of philosophical thought from ancient times to the present, highlighting key thinkers and ideas.",
    "Classical Archaeology: Exploring the Ancient Mediterranean: Investigating the archaeological remains of ancient Greece and Rome, understanding their cultures and civilizations.",
    "Comparative Linguistics: Languages in Contrast: Examining linguistic structures across languages, understanding language families, and exploring language change and evolution.",
    "Criminology: The Study of Crime and Society: Investigating the causes, prevention, and control of crime, with insights into criminal behavior and justice systems.",
    "Business Ethics: Principles and Case Studies: Exploring ethical challenges in business practices, corporate responsibility, and ethical decision-making models.",
    "Industrial Organization: Markets, Strategies, and Performance: Analyzing the structure of industries, firm behavior, and market competition, focusing on strategic interactions.",
    "Gastroenterology: Digestive Diseases and Treatment: Studying the digestive system, its disorders, and the medical and surgical treatments available.",
    "Biblical Studies: Texts and Interpretations: Examining the texts of the Bible, their historical context, theological significance, and interpretive traditions.",
    "Entomology: The Fascinating World of Insects: Exploring insect biology, diversity, ecology, and their roles in ecosystems and human society.",
    "Public Policy: Creating Change in Society: Analyzing the development, implementation, and impact of public policies on communities and societies.",
    "Papyrology: Ancient Documents Unveiled: Studying ancient papyrus documents to uncover insights into the languages, cultures, and daily life of past civilizations.",
    "Classical Drama: Theater of Ancient Greece and Rome: Exploring the origins, characteristics, and influence of ancient Greek and Roman drama.",
    "Geophysics: Earth's Dynamics Explored: Investigating the physical processes governing the Earth, including seismic activity, magnetic and gravitational fields.",
    "Philosophy of Mathematics: Nature and Knowledge: Examining foundational issues in mathematics, including the nature of mathematical objects, truth, and proof.",
    "Latin Literature: Masterpieces and Influence: Studying key works of Latin literature, their historical context, literary forms, and enduring impact.",
    "Ancient Near Eastern Studies: Civilizations and Legacies: Investigating the history, languages, literatures, and cultures of the ancient Near East.",
    "African American Studies: History, Culture, and Politics: Exploring the experiences, contributions, and challenges of African Americans throughout history.",
    "Philosophy of History: Interpreting the Past: Analyzing theories and methodologies for studying and interpreting historical events and narratives.",
    "Metallurgy: The Science of Metals: Understanding the properties, processing, and applications of metals and alloys in various industries.",
    "Ancient Philosophy: Beginnings of Western Thought: Investigating the philosophical ideas of ancient Greek and Roman thinkers and their influence on Western philosophy.",
    "Comparative Education: Global Perspectives: Analyzing educational systems, policies, and practices worldwide, focusing on comparative approaches and outcomes.",
    "Philosophy of Technology: Critique and Conception: Examining the philosophical implications of technology, its impact on society, and ethical considerations.",
    "Gerontology: Aging in Society: Studying the biological, psychological, and social aspects of aging, including health care, policy, and quality of life.",
    "Rural Sociology: Community, Agriculture, and Change: Analyzing social structures, institutions, and dynamics in rural settings, with a focus on agricultural communities.",
    "Ancient Greek Literature: Epics, Dramas, and Philosophy: Exploring the literary and cultural contributions of ancient Greece, including epic poetry, drama, and philosophical texts.",
    "Experimental Psychology: Methods and Discoveries: Investigating psychological phenomena through experimental methods, focusing on design, data analysis, and interpretation.",
    "Kinesiology: Movement and Performance: Examining the scientific principles underlying human movement, including biomechanics, motor control, and physical activity.",
    "Judicial Studies: Courts and the Law: Exploring the functioning of judicial systems, legal reasoning, and the role of courts in society.",
    "Economic History: Markets and Societies Over Time: Tracing the development of economies over time, including trade, industrialization, and financial systems.",
    "Social Psychology: Behavior in Context: Understanding how individual behavior is influenced by social environments, including group dynamics, persuasion, and social perception.",
    "Political Sociology: Power and Participation: Examining the interplay between society and politics, including power structures, political behavior, and social movements.",
    "Cultural Studies: Media, Identity, and Power: Analyzing cultural practices, media representations, and their impacts on identity, society, and politics."
]

# export OPENAI_API_KEY=sk-...
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Constants
CONCURRENT_CALLS = 10
DATASET_PATH = "dialogs.pkl"
FETCH_TIMEOUT = 30
RETRY_ATTEMPTS = 10

def fetch_dialog_with_retry():
    for attempt in range(RETRY_ATTEMPTS):
        try:
            prompt = f"Write a socratic dialog, in which the user begins, between an assistant and his user about {random.choice(curriculum)}"
            response = client.chat.completions.create(model="gpt-3.5-turbo-0125",  messages=[{"role": "user", "content": prompt}], temperature=1.0, max_tokens=2048)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error fetching dialog (Attempt {attempt + 1}/{RETRY_ATTEMPTS}): {e}\nCooling down for {(attempt + 2) * 2} seconds before retrying...")
            time.sleep((attempt + 2) * 2)
    return None

def generate_dialogs(file_path):
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
    responses = []
    
    with ThreadPoolExecutor() as executor:
        tasks = [executor.submit(fetch_dialog_with_retry) for _ in range(CONCURRENT_CALLS)]
        for task in tqdm(as_completed(tasks), total=CONCURRENT_CALLS, desc=f"Fetching dialogs"):
            response = None
            try:
                response = task.result(timeout=FETCH_TIMEOUT)
            except TimeoutError:
                print("Task exceeded the allowed time to complete.")
            if response:
                responses.append(response)


    updated_responses = []
    for response in responses:
        try:
            response = response.lower()

            if response.count("user: ") == 0 or response.count("\n\nuser: ") <= 1 or response.count("\n\nassistant: ") <= 1 or response.count("socratic dialog") >= 1:
                print(f"Ill-formed dialog. Skipping...")
                continue
            
            response = response.replace("\n\nuser: ", "\n[USER] ")
            response = response.replace("\n\nassistant: ", "\n[ASSISTANT] ")
            response = response.replace("user: ", "[USER] ")
            response = response.replace("\n", " ")
            updated_responses.append(response)
        except Exception as e:
            print(f"Error processing response: {e}\nSkipping...")
            continue

    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            existing_responses = pickle.load(file)
        updated_responses.extend(existing_responses)
    
    with open(file_path, "wb") as file:
        pickle.dump(updated_responses, file)

if __name__ == "__main__":
    while True:
        generate_dialogs(DATASET_PATH)
        print("Cooling down for 30 seconds...")
        time.sleep(30)