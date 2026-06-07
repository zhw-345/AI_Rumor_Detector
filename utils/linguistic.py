'''from langchain_core.documents import Document
from langchain_chroma import Chroma
import os

KNOWLEDGE_DB_PATH = "./chroma_pheme_knowledge"

def build_knowledge_db(persist_dir: str):
    """构建谣言检测领域知识库"""
    
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print(f"加载已有知识库: {persist_dir}")
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
    
    print(f"构建新知识库: {persist_dir}")
    
    all_knowledge = []

    linguistic_knowledge = [
    Document(
        page_content="""
RUMOR LINGUISTIC PATTERN - Absolute Language:
Rumors frequently use absolute terms to create false certainty. Examples include:
- "100% effective", "guaranteed cure", "always works"
- "everyone knows", "nobody is talking about this"
- "this is the only solution"

These phrases eliminate nuance and discourage critical thinking. Non-rumors typically use qualified language: "studies suggest", "according to experts", "preliminary data indicates".
        """,
        metadata={"category": "linguistic_pattern", "subcategory": "absolute_language", "source": "expert_curated"}
    ),
    
    Document(
        page_content="""
RUMOR LINGUISTIC PATTERN - Emotional Manipulation:
Rumors exploit emotional triggers through:
- FEAR: "urgent warning", "protect your family immediately", "before it's too late"
- OUTRAGE: "this is disgusting", "they don't want you to know", "share before they delete"
- FALSE HOPE: "miracle breakthrough", "what doctors don't want you to know"

Non-rumors report facts without emotional coercion.
        """,
        metadata={"category": "linguistic_pattern", "subcategory": "emotional_manipulation", "source": "expert_curated"}
    ),
    
    Document(
        page_content="""
RUMOR LINGUISTIC PATTERN - False Attribution:
Rumors frequently attribute claims to unnamed or unverifiable sources:
- "a friend who works in the government told me"
- "many experts are saying"
- "studies show" (without citation)
- "people are reporting"

Non-rumors cite specific, verifiable sources: "According to a study published in Nature (2024)...", "Dr. Smith at Johns Hopkins confirmed..."
        """,
        metadata={"category": "linguistic_pattern", "subcategory": "false_attribution", "source": "expert_curated"}
    ),
]
    
    # === 1. 语言学模式 ===
    linguistic_patterns = [
        Document(
            page_content="RUMOR PATTERN - Absolute Language: Rumors use '100%', 'guaranteed', 'always', 'everyone knows' to create false certainty. Non-rumors use qualified language: 'studies suggest', 'according to experts'.",
            metadata={"category": "linguistic", "pattern": "absolute_language", "priority": "high"}
        ),
        Document(
            page_content="RUMOR PATTERN - Emotional Manipulation: Fear triggers ('urgent warning', 'protect your family'), outrage triggers ('they don't want you to know'), false hope ('miracle breakthrough'). Non-rumors report facts without coercion.",
            metadata={"category": "linguistic", "pattern": "emotional_manipulation", "priority": "high"}
        ),
        Document(
            page_content="RUMOR PATTERN - False Attribution: 'a friend in government', 'many experts', 'studies show' without citation. Non-rumors cite specific verifiable sources.",
            metadata={"category": "linguistic", "pattern": "false_attribution", "priority": "high"}
        ),
        Document(
            page_content="RUMOR PATTERN - Urgency and Scarcity: 'share before deleted', 'limited time', 'breaking', 'just leaked'. These create artificial pressure to bypass critical evaluation.",
            metadata={"category": "linguistic", "pattern": "urgency_scarcity", "priority": "medium"}
        ),
        Document(
            page_content="RUMOR PATTERN - Conspiracy Framing: 'mainstream media won't report', 'what they don't want you to know', 'the truth about'. Positions the speaker as possessing secret knowledge.",
            metadata={"category": "linguistic", "pattern": "conspiracy_framing", "priority": "medium"}
        ),
    ]
    all_knowledge.extend(linguistic_patterns)
    
    # === 2. 主题领域知识 ===
    domain_knowledge = [
        Document(
            page_content="DOMAIN - Terrorism Events: Rumors misidentify suspects, inflate casualties, claim secondary attacks, circulate hero narratives. Verification: official police statements, multiple news agencies.",
            metadata={"category": "domain", "domain": "terrorism", "events": "charliehebdo,ottawashooting,sydneysiege"}
        ),
        Document(
            page_content="DOMAIN - Aviation Disasters: Rumors speculate causes (mechanical/terrorism/pilot error), claim survivors in impossible conditions, spread conspiracy theories ('shot down', 'hijacked'). Verification: BEA, NTSB, black box transcripts.",
            metadata={"category": "domain", "domain": "aviation", "events": "germanwings-crash"}
        ),
        Document(
            page_content="DOMAIN - Art/Provenance: Rumors forge ownership history, inflate values, dispute authenticity without expert review, obstruct restitution. Reliable indicators: named museums, specific curators, provenance research institutions.",
            metadata={"category": "domain", "domain": "art_provenance", "events": "gurlitt"}
        ),
        Document(
            page_content="DOMAIN - Social Unrest: Rumors exaggerate violence, misattribute actions to groups, circulate fake images from other events. Verification: geolocation, image reverse search, timestamp analysis.",
            metadata={"category": "domain", "domain": "social_unrest", "events": "ferguson"}
        ),
    ]
    all_knowledge.extend(domain_knowledge)
    
    # === 3. 事实核查标准 ===
    fact_checking = [
        Document(
            page_content="FACT-CHECKING - Source Hierarchy: Tier 1 (primary/official), Tier 2 (established news), Tier 3 (credentialed experts), Tier 4 (eyewitness), Tier 5 (anonymous social media). Rumors claim Tier 1-2 authority while using Tier 4-5 sources.",
            metadata={"category": "fact_checking", "pattern": "source_hierarchy"}
        ),
        Document(
            page_content="FACT-CHECKING - Temporal Signatures: Early phase (0-2h): high volume, low accuracy. Amplification (2-6h): false narratives solidify. Correction (6-24h): official statements emerge. Residual (24h+): rumors persist in fringe communities only.",
            metadata={"category": "fact_checking", "pattern": "temporal_analysis"}
        ),
        Document(
            page_content="FACT-CHECKING - Image Verification: Check EXIF data, reverse image search, geolocation landmarks, shadow analysis for time-of-day consistency. Rumors frequently reuse images from unrelated events.",
            metadata={"category": "fact_checking", "pattern": "image_verification"}
        ),
    ]
    all_knowledge.extend(fact_checking)
    
    # === 4. 数据集标注标准 ===
    annotation_standards = [
        Document(
            page_content="PHEME RUMOR CRITERIA: Unverified claim as fact, lacks credible source, speculative/emotional language, contradicted by official info, rapid spread. NON-RUMOR: Verified facts, qualified language, specific checkable details, aligns with official info.",
            metadata={"category": "annotation", "dataset": "PHEME"}
        ),
    ]
    all_knowledge.extend(annotation_standards)
    
    # 构建向量库
    vectorstore = Chroma.from_documents(
        documents=all_knowledge,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    print(f"知识库构建完成: {len(all_knowledge)} 条知识")
    return vectorstore


# 执行构建
knowledge_db = build_knowledge_db(KNOWLEDGE_DB_PATH)

# 配置检索器
knowledge_retriever = knowledge_db.as_retriever(
    search_type="similarity",  # 知识库用纯相似度即可，知识条目不重复
    search_kwargs={"k": 3}
)'''

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

KNOWLEDGE_DB_PATH = "./chroma_linguistic_features"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Embedding 初始化
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)


def build_linguistic_feature_db(persist_dir: str):
    """Build linguistic feature knowledge base for rumor detection"""
    
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print(f"Loading existing database: {persist_dir}")
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
    
    print(f"Building new database: {persist_dir}")
    
    all_knowledge = []

    # === 1. QUANTITY FEATURES ===
    quantity_features = [
        Document(
            page_content="""
LINGUISTIC FEATURE - Word Count:
Deceptive texts exhibit significantly lower word counts (mean: 62 words) compared to truthful texts (mean: 81 words). In social media contexts, fake news tweets are longer (36.32 vs 34.07 words) with more characters (223.5 vs 211.7). 
Rule: Below-average word count in long-form text suggests deception; above-average length in tweets suggests fake news.
            """,
            metadata={"category": "quantity", "feature": "word_count", "priority": "high", "cross_cultural": "true"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Sentence Count:
Deceptive statements contain significantly fewer sentences than truthful statements. Truthful narratives use more sentences to provide comprehensive information.
Rule: Unusually low sentence count relative to topic complexity indicates potential deception.
            """,
            metadata={"category": "quantity", "feature": "sentence_count", "priority": "high", "cross_cultural": "true"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Verb Count:
Fake news and deceptive texts use fewer verbs, reducing action descriptions and concrete event reporting.
Rule: Low verb density suggests avoidance of specific action claims.
            """,
            metadata={"category": "quantity", "feature": "verb_count", "priority": "medium", "cross_cultural": "true"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Average Sentence Length:
Longer sentences indicate higher credibility writing. Deceptive texts tend toward shorter sentences, though dataset variance exists.
Rule: Significantly shorter average sentence length than genre baseline suggests simplified or evasive construction.
            """,
            metadata={"category": "quantity", "feature": "avg_sentence_length", "priority": "medium", "cross_cultural": "false"}
        ),
    ]
    all_knowledge.extend(quantity_features)

    # === 2. COMPLEXITY & READABILITY FEATURES ===
    complexity_features = [
        Document(
            page_content="""
LINGUISTIC FEATURE - Flesch Reading Ease:
Scale: 90-100 (5th grade, very easy), 80-90 (6th grade, easy), 70-80 (7th grade, fairly easy), 60-70 (8th-9th grade, standard), 50-60 (10th-12th grade, fairly difficult), 30-50 (college, difficult), 10-30 (graduate, very difficult), 0-10 (professional, extremely difficult).
Fake news tweets score 55.67 vs 53.05 for real news (slightly easier). Direction varies by dataset.
Rule: Extreme deviation from genre-appropriate readability level warrants investigation.
            """,
            metadata={"category": "complexity", "feature": "flesch_reading_ease", "priority": "medium", "scale": "0-100"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Flesch-Kincaid Grade Level:
Grade levels: 0-5 (elementary), 6-8 (middle school), 9-10 (high school), 11-12 (college), 13-16 (graduate), 17+ (professional).
Rule: Texts targeting sophisticated audiences but scoring below 9th grade may indicate simplified deception; texts scoring above 16 for general audiences may indicate obfuscation.
            """,
            metadata={"category": "complexity", "feature": "flesch_kincaid_grade", "priority": "medium", "scale": "grade_level"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Gunning Fog Index:
Calculates grade level based on percentage of complex words (>=3 syllables). Grade 12 is the threshold for general business writing.
Rule: Scores significantly above 12 for general news, or significantly below 8 for academic topics, indicate stylistic inconsistency.
            """,
            metadata={"category": "complexity", "feature": "gunning_fog", "priority": "low", "threshold": "grade_12"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Complex Word Ratio:
Proportion of words with 3+ syllables. Fake news headlines use fewer nouns and more proper nouns compared to real news.
Rule: Abnormal proper noun density in headlines may indicate source fabrication.
            """,
            metadata={"category": "complexity", "feature": "complex_word_ratio", "priority": "low", "cross_cultural": "false"}
        ),
    ]
    all_knowledge.extend(complexity_features)

    # === 3. PUNCTUATION & STYLE FEATURES ===
    punctuation_features = [
        Document(
            page_content="""
LINGUISTIC FEATURE - Exclamation Density:
Fake news tweets use 56% more exclamation marks (0.309 vs 0.198 density). Repeated exclamation marks (!!) are indicators of health-related rumors.
Rule: Exclamation density >0.25 in news text suggests emotional manipulation.
            """,
            metadata={"category": "punctuation", "feature": "exclamation_density", "priority": "high", "threshold": "0.25"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Question Density:
Fake news uses more question marks. HARD THRESHOLD: Inquiry ratio (questioning comments / total comments) >= 0.1 indicates rumor; < 0.1 indicates non-rumor.
Rule: High question density in claims or comments suggests unverified information seeking validation.
            """,
            metadata={"category": "punctuation", "feature": "question_density", "priority": "high", "hard_threshold": "0.1"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Repeated Punctuation:
Consecutive repeated punctuation (e.g., !!, ??, !?!, ...) is a significant indicator in health rumors.
Rule: Presence of repeated punctuation clusters indicates emotional agitation or artificial urgency.
            """,
            metadata={"category": "punctuation", "feature": "repeated_punctuation", "priority": "medium", "domain": "health"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - All-Caps Ratio:
Fake news contains more all-capitalized words used for emphasis and sensationalism.
Rule: All-caps word ratio >0.05 in formal news text indicates shouting/sensationalism.
            """,
            metadata={"category": "punctuation", "feature": "all_caps_ratio", "priority": "medium", "threshold": "0.05"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Quotation Usage:
Contrary to credibility-transfer hypothesis, real news uses MORE quotation marks than fake news in most datasets.
Rule: Absence of direct quotes in reported speech suggests source fabrication.
            """,
            metadata={"category": "punctuation", "feature": "quotation_usage", "priority": "medium", "cross_cultural": "false"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Repetition Rate:
Fake news shows higher repetition rate (0.102 vs 0.094, +8.1%). Fake news headlines are longer with fewer stopwords.
Rule: Repetition rate >10% suggests keyword stuffing or circular reasoning.
            """,
            metadata={"category": "punctuation", "feature": "repetition_rate", "priority": "medium", "threshold": "0.10"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Hashtag Count:
Fake news tweets use 35.7% fewer hashtags than real news.
Rule: Unusually low hashtag usage in social media news may indicate avoidance of discoverability/verification.
            """,
            metadata={"category": "punctuation", "feature": "hashtag_count", "priority": "low", "platform": "twitter"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Number Ratio:
Fake news uses 16.6% fewer numbers (0.014 vs 0.017 density).
Rule: Absence of quantifiable data in data-driven claims suggests fabrication.
            """,
            metadata={"category": "punctuation", "feature": "number_ratio", "priority": "medium", "cross_cultural": "true"}
        ),
    ]
    all_knowledge.extend(punctuation_features)

    # === 4. AFFECT & PSYCHOLOGICAL FEATURES (LIWC) ===
    affect_features = [
        Document(
            page_content="""
LINGUISTIC FEATURE - Negative Emotion (LIWC):
TOP-RANKED deception indicator. Fake news and deceptive statements show significantly higher negative emotion word density.
Rule: Negative emotion density in top quartile of genre baseline strongly suggests deception.
            """,
            metadata={"category": "affect", "feature": "negative_emotion", "priority": "high", "liwc": "true", "rank": "1"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Anger (LIWC Anger):
#1 LIWC feature for deception detection. Political fake news shows elevated anger word usage.
Rule: Anger word density >2x genre median indicates outrage manipulation.
            """,
            metadata={"category": "affect", "feature": "anger", "priority": "high", "liwc": "true", "rank": "1"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Anxiety (LIWC Anxiety):
Significantly elevated in health-related rumors and pandemic misinformation.
Rule: Anxiety word density above genre 90th percentile in health contexts indicates fearmongering.
            """,
            metadata={"category": "affect", "feature": "anxiety", "priority": "high", "liwc": "true", "domain": "health"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Sadness (LIWC Sadness):
Fake news uses more sadness expressions, particularly in sympathy-seeking narratives.
Rule: Sadness density disproportionate to topic baseline suggests emotional exploitation.
            """,
            metadata={"category": "affect", "feature": "sadness", "priority": "medium", "liwc": "true"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Positive Emotion (LIWC Posemo):
Truthful statements use more optimism and friendship words. Fake news shows reduced positive emotion.
Rule: Positive emotion density in bottom quartile suggests pessimistic/deceptive framing.
            """,
            metadata={"category": "affect", "feature": "positive_emotion", "priority": "medium", "liwc": "true", "cross_cultural": "true"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Certainty (LIWC Certain):
Deceptive texts use more absolute certainty words ("never", "always", "absolutely"). Cross-culturally observed in Spanish deception.
Rule: Certainty word density >2x baseline combined with negation density suggests false confidence.
            """,
            metadata={"category": "affect", "feature": "certainty", "priority": "high", "liwc": "true", "cross_cultural": "true"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Negation (LIWC Negate):
Cross-culturally robust: all languages show deceptive texts use MORE negation words ("no", "not", "never").
Rule: Negation density in top quartile is a strong deception marker across all languages.
            """,
            metadata={"category": "affect", "feature": "negation", "priority": "high", "liwc": "true", "cross_cultural": "true"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Tentative (LIWC Tentat):
Top-10 deception feature. Deceptive texts may use more hedges ("possibly", "maybe") or fewer depending on strategy.
Rule: Extreme values (very high or very low) relative to baseline suggest deliberate linguistic strategy.
            """,
            metadata={"category": "affect", "feature": "tentative", "priority": "medium", "liwc": "true", "rank": "top10"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Nonfluencies (LIWC Nonflu):
Top-10 deception feature. Filler words ("um", "uh", "like") indicate cognitive load in spoken deception.
Rule: High nonfluency density in written text suggests simulated spontaneity or translated speech.
            """,
            metadata={"category": "affect", "feature": "nonfluencies", "priority": "low", "liwc": "true", "rank": "top10", "modality": "spoken"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Religion (LIWC Relig):
Top-10 deception feature in political contexts. Religious language used to establish moral authority.
Rule: Religious word density above baseline in non-religious topics indicates authority manipulation.
            """,
            metadata={"category": "affect", "feature": "religion", "priority": "low", "liwc": "true", "rank": "top10", "domain": "political"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Motion (LIWC Motion):
Top-10 deception feature. Deceptive texts use more motion words ("go", "run", "move").
Rule: Excessive motion word density may indicate narrative distraction or fabricated action.
            """,
            metadata={"category": "affect", "feature": "motion", "priority": "low", "liwc": "true", "rank": "top10", "cross_cultural": "true"}
        ),
    ]
    all_knowledge.extend(affect_features)

    # === 5. PRONOUN & SELF-REFERENCE FEATURES ===
    pronoun_features = [
        Document(
            page_content="""
LINGUISTIC FEATURE - First Person Singular (I/me/my):
Cross-culturally robust: truthful statements use MORE first-person singular. Deceptive texts reduce self-reference to avoid accountability.
Rule: First-person singular density in bottom quartile of personal narrative suggests distancing.
            """,
            metadata={"category": "pronoun", "feature": "first_person_singular", "priority": "high", "liwc": "true", "cross_cultural": "true"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - First Person Plural (We/us/our):
Deceptive texts may use more plural pronouns to create false consensus or group identity.
Rule: High "we" usage in individual claims suggests manufactured solidarity.
            """,
            metadata={"category": "pronoun", "feature": "first_person_plural", "priority": "medium", "liwc": "true"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Second Person (You/your):
Top-10 deception feature. Deceptive texts directly address readers more frequently for persuasion.
Rule: Second-person density >2x baseline in non-interactive contexts indicates targeted manipulation.
            """,
            metadata={"category": "pronoun", "feature": "second_person", "priority": "medium", "liwc": "true", "rank": "top10", "cross_cultural": "true"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Third Person Pronouns (He/she/they):
Cross-culturally robust: deceptive texts use MORE third-person pronouns to deflect focus from self.
Rule: Third-person density in top quartile combined with low first-person suggests deflection.
            """,
            metadata={"category": "pronoun", "feature": "third_person", "priority": "high", "liwc": "true", "cross_cultural": "true"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Personal Pronoun Density (PRP):
Top-10 POS deception feature. Overall pronoun usage pattern anomalies indicate deception.
Rule: Unusual pronoun distribution (high third-person, low first-person) is a strong deception signature.
            """,
            metadata={"category": "pronoun", "feature": "pronoun_density", "priority": "medium", "pos": "PRP", "rank": "top10"}
        ),
    ]
    all_knowledge.extend(pronoun_features)

    # === 6. COGNITIVE PROCESS FEATURES ===
    cognitive_features = [
        Document(
            page_content="""
LINGUISTIC FEATURE - Insight Words (LIWC Insight):
Cognitive process words ("know", "understand", "discover") show significant differences between true and false texts.
Rule: Low insight word density in analytical claims suggests superficial understanding or fabrication.
            """,
            metadata={"category": "cognitive", "feature": "insight", "priority": "medium", "liwc": "true"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Causal Words (LIWC Cause):
Words indicating causality ("because", "so", "lead to"). Fake news shows confused or oversimplified causal logic.
Rule: Absence of causal connectors in explanatory text suggests evasion of logical accountability.
            """,
            metadata={"category": "cognitive", "feature": "causal", "priority": "medium", "liwc": "true"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Memory Words (LIWC Memory):
Words related to memory ("remember", "recall", "used to"). True narratives contain more memory references.
Rule: Low memory word density in personal narratives suggests fabricated experience.
            """,
            metadata={"category": "cognitive", "feature": "memory", "priority": "medium", "liwc": "true", "cross_cultural": "true"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Time Words (LIWC Time):
Top-10 deception feature. Deceptive texts may use more vague temporal references.
Rule: Temporal specificity (specific dates vs. "recently", "lately") distinguishes true from false reports.
            """,
            metadata={"category": "cognitive", "feature": "time", "priority": "medium", "liwc": "true", "rank": "top10"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Space Words (LIWC Space):
Spatial references ("here", "there", "above"). Deceptive texts reduce spatial detail.
Rule: Low spatial specificity in event descriptions suggests lack of genuine presence.
            """,
            metadata={"category": "cognitive", "feature": "space", "priority": "low", "liwc": "true", "cross_cultural": "true"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Perceptual Words (LIWC See/Hear/Felt):
Cross-culturally robust: truthful texts use MORE sensory words. Deceptive texts reduce sensory detail.
Rule: Absence of sensory specificity in eyewitness claims suggests fabrication.
            """,
            metadata={"category": "cognitive", "feature": "perceptual", "priority": "high", "liwc": "true", "cross_cultural": "true"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Cognitive Load Indicators:
Complex clause nesting, self-corrections, and syntactic complexity indicate increased cognitive load during deception.
Rule: Unusual syntactic complexity (either excessive or notably simple) may indicate deception strategy.
            """,
            metadata={"category": "cognitive", "feature": "cognitive_load", "priority": "low", "modality": "spoken"}
        ),
    ]
    all_knowledge.extend(cognitive_features)

    # === 7. VAGUENESS & SPECIFICITY FEATURES ===
    specificity_features = [
        Document(
            page_content="""
LINGUISTIC FEATURE - Vague Terms (LIWC):
Generalization words ("everyone", "always", "some", "certain people"). Deceptive texts use more vague terms to avoid falsifiability.
Rule: Vague term density in top quartile combined with low specific term density suggests evasion.
            """,
            metadata={"category": "specificity", "feature": "vague_terms", "priority": "high", "liwc": "true", "cross_cultural": "true"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Hedges (LIWC):
Hedging words ("possibly", "I think", "sort of"). Deceptive texts may use more hedges to avoid commitment or fewer to project false confidence.
Rule: Extreme hedge density (very high or very low) relative to genre suggests deliberate strategy.
            """,
            metadata={"category": "specificity", "feature": "hedges", "priority": "medium", "liwc": "true"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Specificity Score:
Verifiable concrete details vs. abstract claims. Fake news scores lower on specificity.
Rule: Specificity score in bottom quartile for factual claims suggests fabrication.
            """,
            metadata={"category": "specificity", "feature": "specificity_score", "priority": "high", "cross_cultural": "true"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Unique Word Count:
Fake news tweets show more unique words (32.12 vs 30.36) but lower overall vocabulary diversity (TTR).
Rule: High unique word count with low TTR suggests keyword variation without semantic depth.
            """,
            metadata={"category": "specificity", "feature": "unique_words", "priority": "low", "platform": "twitter"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Vocabulary Diversity (TTR/MATTR):
Type-Token Ratio or Moving Average TTR. Deceptive texts show reduced vocabulary diversity.
Rule: MATTR below genre 20th percentile suggests limited linguistic repertoire or formulaic generation.
            """,
            metadata={"category": "specificity", "feature": "vocabulary_diversity", "priority": "medium", "cross_cultural": "true"}
        ),
    ]
    all_knowledge.extend(specificity_features)

    # === 8. SYNTACTIC & POS FEATURES ===
    syntactic_features = [
        Document(
            page_content="""
LINGUISTIC FEATURE - Wh-Adverbs (WRB):
Top-10 POS deception feature ("how", "why", "when"). Deceptive texts show abnormal question-word patterns.
Rule: High WRB density in declarative text suggests embedded questioning or evasion.
            """,
            metadata={"category": "syntactic", "feature": "wh_adverbs", "priority": "low", "pos": "WRB", "rank": "top10"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Gerunds (VBG):
Top-10 POS deception feature. Deceptive texts show abnormal gerund usage patterns.
Rule: Unusual gerund density may indicate nominalization strategy to obscure agency.
            """,
            metadata={"category": "syntactic", "feature": "gerunds", "priority": "low", "pos": "VBG", "rank": "top10"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Comparative Adjectives (JJR):
Top-10 POS deception feature ("bigger", "better"). Deceptive texts use more comparatives for exaggeration.
Rule: Comparative density >2x baseline suggests competitive/framing manipulation.
            """,
            metadata={"category": "syntactic", "feature": "comparative_adj", "priority": "low", "pos": "JJR", "rank": "top10"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Superlative Adjectives (JJS):
Top-10 POS deception feature ("biggest", "best"). Deceptive texts use more superlatives for absolutization.
Rule: Superlative density in top decile suggests hyperbolic claims.
            """,
            metadata={"category": "syntactic", "feature": "superlative_adj", "priority": "low", "pos": "JJS", "rank": "top10"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Proper Nouns (NNP):
Top-10 POS deception feature. Fake news headlines contain more proper nouns.
Rule: Excessive proper noun density without corresponding specific details suggests name-dropping.
            """,
            metadata={"category": "syntactic", "feature": "proper_nouns", "priority": "medium", "pos": "NNP", "rank": "top10"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Interjections (UH):
Top-10 POS deception feature ("oh", "wow", "um"). More frequent in spoken or simulated-spoken deception.
Rule: High interjection density in written formal text suggests simulated authenticity.
            """,
            metadata={"category": "syntactic", "feature": "interjections", "priority": "low", "pos": "UH", "rank": "top10", "modality": "spoken"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Particles (RP):
Top-10 POS deception feature ("up", "out", "off"). Phrasal verb usage anomalies in deceptive text.
Rule: Unusual particle distribution may indicate non-native generation or deliberate obfuscation.
            """,
            metadata={"category": "syntactic", "feature": "particles", "priority": "low", "pos": "RP", "rank": "top10"}
        ),
    ]
    all_knowledge.extend(syntactic_features)

    # === 9. INQUIRY & INTERACTION FEATURES ===
    inquiry_features = [
        Document(
            page_content="""
LINGUISTIC FEATURE - Inquiry Ratio:
HARD THRESHOLD: Inquiry ratio (questioning comments / total English comments) >= 0.1 indicates RUMOR; < 0.1 indicates NON-RUMOR. Based on Snopes-verified dataset.
Rule: This is one of the few validated hard thresholds in rumor detection.
            """,
            metadata={"category": "inquiry", "feature": "inquiry_ratio", "priority": "high", "hard_threshold": "0.1", "source": "snopes"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Reply-to-Retweet Ratio:
Abnormal interaction patterns where replies disproportionately exceed retweets may indicate controversial/unverified content.
Rule: Reply/retweet ratio >3x account baseline suggests audience skepticism.
            """,
            metadata={"category": "inquiry", "feature": "reply_retweet_ratio", "priority": "low", "platform": "twitter"}
        ),
        Document(
            page_content="""
LINGUISTIC FEATURE - Comment Sentiment Polarity:
Rumor comments show more dispersed or negative sentiment polarity compared to verified news.
Rule: High sentiment variance in comments suggests audience uncertainty.
            """,
            metadata={"category": "inquiry", "feature": "comment_sentiment", "priority": "low", "platform": "social_media"}
        ),
    ]
    all_knowledge.extend(inquiry_features)

    # === 10. CROSS-CULTURALLY ROBUST FEATURES ===
    cross_cultural_features = [
        Document(
            page_content="""
CROSS-CULTURAL ROBUST FEATURES:
The following features show consistent patterns across English, Hindi, and Spanish deception detection:
- Word Count: deceptive texts are shorter (62 vs 81 words)
- Sentence Count: deceptive texts have fewer sentences
- Verb Count: deceptive texts use fewer verbs
- First Person Singular: truthful texts use more (I/me/my)
- Third Person Pronouns: deceptive texts use more (he/she/they)
- Negation: deceptive texts use more negation words across all languages
- Certainty Words: Spanish deceptive texts use more ("never", "always")
- Perceptual Words: truthful texts use more sensory words (see/hear/felt)
These features should be prioritized for multilingual rumor detection systems.
            """,
            metadata={"category": "cross_cultural", "feature": "robust_set", "priority": "high", "languages": "english,hindi,spanish"}
        ),
    ]
    all_knowledge.extend(cross_cultural_features)

    # Build vector store
    vectorstore = Chroma.from_documents(
        documents=all_knowledge,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    print(f"Database built: {len(all_knowledge)} linguistic features")
    return vectorstore


# Execute build
linguistic_db = build_linguistic_feature_db(KNOWLEDGE_DB_PATH)

# Configure retriever
linguistic_retriever = linguistic_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

test = "Fake news usually use less words."
results = linguistic_retriever.invoke(test)
for result in results:
    print(result)