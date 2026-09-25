from typing import Optional, List, Tuple, Dict, Iterable, Union

from flashrank import Ranker, RerankRequest
from ovos_plugin_manager.templates.solvers import TldrSolver, MultipleChoiceSolver, QuestionSolver, EvidenceSolver
from ovos_utils.log import LOG
from quebra_frases import sentence_tokenize, paragraph_tokenize


class FlashRankMultipleChoiceSolver(MultipleChoiceSolver):
    """select best answer to a question from a list of options """

    def __init__(self, config=None):
        config = config or {"min_conf": None,
                            "n_answer": 1,
                            "model": "ms-marco-MultiBERT-L-12"}
        super().__init__(config)
        self.ranker = Ranker(model_name=self.config.get("model", "ms-marco-MultiBERT-L-12"))

    def rerank(self, query: str, options: List[str],
               lang: Optional[str] = None,
               return_index: bool = False) -> List[Tuple[float, Union[str, int]]]:
        """
        rank options list, returning a list of tuples (score, text)
        """
        passages = [
            {"text": o}
            for o in options
        ]
        rerankrequest = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerankrequest)
        # min_conf has been in this plugin's default config since the first
        # commit and nothing read it. The model always ranks, so without a
        # floor the caller is handed a best option even when the query matches
        # nothing in the list. Measured on ms-marco-MultiBERT-L-12, three
        # unrelated options: a query the list answers scores 0.9972 and 0.9993,
        # a query it does not scores 0.6397 and 0.3120, and gibberish scores
        # 0.0012. The gap is wide, and where to cut it depends on how a
        # deployment would rather fail, so the default stays None, which is
        # the behaviour this plugin has always had.
        min_conf = self.config.get("min_conf")
        if min_conf:
            results = [r for r in results if r["score"] >= min_conf]
        if return_index:
            return [(r["score"], options.index(r["text"])) for r in results]
        return [(r["score"], r["text"]) for r in results]

    def select_answer(self, query: str, options: List[str],
                      lang: Optional[str] = None,
                      return_index: bool = False) -> Optional[Union[str, int]]:
        """
        Return the best option, or None when none of them clears min_conf.

        The base class takes the first result unconditionally, which raises
        IndexError once the floor above empties the list. None says "no answer"
        instead, which is what a caller can act on.
        """
        ranked = self.rerank(query, options, lang=lang, return_index=return_index)
        if not ranked:
            LOG.debug(f"no option scored at or above min_conf="
                      f"{self.config.get('min_conf')} for query: {query}")
            return None
        return ranked[0][1]


class FlashRankEvidenceSolverPlugin(EvidenceSolver):
    """extract best sentence from text that answers the question, using flashrank"""

    def __init__(self, config=None):
        config = config or {"min_conf": None,
                            "n_answer": 1,
                            "model": "ms-marco-MultiBERT-L-12"}
        super().__init__(config)
        self.ranker = FlashRankMultipleChoiceSolver(self.config)

    def get_best_passage(self, evidence, question,
                         lang: Optional[str] = None):
        """
        evidence and question assured to be in self.default_lang
         returns summary of provided document
        """
        sents = []
        for s in evidence.split("\n"):
            sents += sentence_tokenize(s)
        sents = [s.strip() for s in sents if s]
        return self.ranker.select_answer(question, sents, lang=lang)


class FlashRankCorpusSolver(QuestionSolver):
    enable_tx = False
    priority = 60

    def __init__(self, config=None):
        config = config or {"min_conf": None,
                            "n_answer": 1,
                            "model": "ms-marco-MultiBERT-L-12"}
        super().__init__(config)
        self.corpus = None
        self.ranker = FlashRankMultipleChoiceSolver(self.config)

    def load_corpus(self, corpus: List[str]):
        self.corpus = corpus

    def retrieve_from_corpus(self, query, k=3) -> Iterable[Tuple[float, str]]:
        yield from self.ranker.rerank(query, self.corpus)

    def get_spoken_answer(self, query: str, context: Optional[dict] = None) -> str:
        if self.corpus is None:
            return None
        # Query the corpus
        answers = [a[1] for a in self.retrieve_from_corpus(query, k=self.config.get("n_answer", 1))]
        if answers:
            return ". ".join(answers[:self.config.get("n_answer", 1)])


class FlashRankQACorpusSolver(FlashRankCorpusSolver):
    def __init__(self, config=None):
        self.answers = {}
        super().__init__(config)

    def load_corpus(self, corpus: Dict):
        self.answers = corpus
        super().load_corpus(list(self.answers.keys()))

    def retrieve_from_corpus(self, query, k=1) -> Iterable[Tuple[float, str]]:
        for score, q in super().retrieve_from_corpus(query, k):
            LOG.debug(f"closest question in corpus: {q}")
            yield score, self.answers[q]


class FlashRankSummarizer(TldrSolver):
    """summarize text using flashrank"""

    def __init__(self, config=None):
        config = config or {"min_conf": None,
                            "n_answer": 1,
                            "model": "ms-marco-MultiBERT-L-12"}
        super().__init__(config)
        self.ranker = FlashRankMultipleChoiceSolver(self.config)

    def get_tldr(self, document: str,
                 lang: Optional[str] = None) -> str:
        """
        Summarize the provided document.

        :param document: The text of the document to summarize, assured to be in the default language.
        :param lang: Optional language code.
        :return: A summary of the provided document.
        """
        strategy = self.config.get("strategy", "multi")
        n = self.config.get("max_sentences", 3)
        if strategy == "multi":
            sents = []
            for s in document.split("\n"):
                sents += paragraph_tokenize(s)
            top_k = [s[1] for s in self.ranker.rerank(document,
                                                      [s.strip() for s in sents if s],
                                                      lang=lang)]
            sents = []
            for p in top_k[:3]:
                sents += sentence_tokenize(p)
            top_k = [s[1] for s in self.ranker.rerank(document, sents, lang=lang)]
        elif strategy == "paragraphs":
            sents = []
            for s in document.split("\n"):
                sents += paragraph_tokenize(s)
            top_k = [s[1] for s in self.ranker.rerank(document,
                                                      [s.strip() for s in sents if s],
                                                      lang=lang)]
        else:
            sents = []
            for s in document.split("\n"):
                sents += sentence_tokenize(s)
            top_k = [s[1] for s in self.ranker.rerank(document,
                                                      [s.strip() for s in sents if s],
                                                      lang=lang)]
        return "\n".join(top_k[:n])


if __name__ == "__main__":
    LOG.set_level("DEBUG")
    s = FlashRankSummarizer()

    query = """The possibility of alien life in the solar system has been a topic of interest for scientists and astronomers for many years. The search for extraterrestrial life has been a major focus of space exploration, with numerous missions and discoveries made in recent years. While there is still no concrete evidence of life beyond Earth, the search for alien life continues to be a fascinating and exciting endeavor.
    One of the most promising areas for the search for alien life is the moons of Jupiter and Saturn. These moons, such as Europa and Enceladus, are believed to have subsurface oceans that could potentially harbor life. The presence of water, a key ingredient for life as we know it, has been detected on these moons, and there are also indications of other necessary elements such as carbon, nitrogen, and oxygen.
    Another area of interest for the search for alien life is the asteroid belt between Mars and Jupiter. This region is home to millions of asteroids, some of which may have the right conditions for life to exist. For example, some asteroids have been found to have water and organic compounds, which are essential for life.
    In addition to the moons and asteroids of the solar system, there are also other potential locations for the search for alien life. For example, there are exoplanets, or planets outside of our solar system, that have been discovered in recent years. Some of these exoplanets are believed to be in the habitable zone, which means they are located in the right distance from their star to potentially have liquid water on their surface.
    Despite the potential for alien life in the solar system, there are still many uncertainties and unknowns. The search for extraterrestrial life is a complex and multifaceted endeavor that requires a combination of scientific research, technological advancements, and exploration. While there is still no concrete evidence of life beyond Earth, the search for alien life continues to be a fascinating and exciting endeavor that holds the potential for groundbreaking discoveries in the future."""

    s.config["strategy"] = "multi"
    print(s.get_tldr(query))
    # Some of these exoplanets are believed to be in the habitable zone, which means they are located in the right distance from their star to potentially have liquid water on their surface.
    # In addition to the moons and asteroids of the solar system, there are also other potential locations for the search for alien life.
    # Despite the potential for alien life in the solar system, there are still many uncertainties and unknowns.
    s.config["strategy"] = "paragraphs"
    print(s.get_tldr(query))
    # Another area of interest for the search for alien life is the asteroid belt between Mars and Jupiter. This region is home to millions of asteroids, some of which may have the right conditions for life to exist. For example, some asteroids have been found to have water and organic compounds, which are essential for life.
    # Despite the potential for alien life in the solar system, there are still many uncertainties and unknowns. The search for extraterrestrial life is a complex and multifaceted endeavor that requires a combination of scientific research, technological advancements, and exploration. While there is still no concrete evidence of life beyond Earth, the search for alien life continues to be a fascinating and exciting endeavor that holds the potential for groundbreaking discoveries in the future.
    # In addition to the moons and asteroids of the solar system, there are also other potential locations for the search for alien life. For example, there are exoplanets, or planets outside of our solar system, that have been discovered in recent years. Some of these exoplanets are believed to be in the habitable zone, which means they are located in the right distance from their star to potentially have liquid water on their surface.
    s.config["strategy"] = "sentences"
    print(s.get_tldr(query))
    # This region is home to millions of asteroids, some of which may have the right conditions for life to exist.
    # Some of these exoplanets are believed to be in the habitable zone, which means they are located in the right distance from their star to potentially have liquid water on their surface.
    # The presence of water, a key ingredient for life as we know it, has been detected on these moons, and there are also indications of other necessary elements such as carbon, nitrogen, and oxygen.

    p = FlashRankMultipleChoiceSolver()
    a = p.rerank("what is the speed of light", [
        "very fast", "10m/s", "the speed of light is C"
    ])
    print(a)
    # [(0.999819, 'the speed of light is C'),
    # (2.7686672e-05, 'very fast'),
    # (1.2555749e-05, '10m/s')]

    a = p.select_answer("what is the speed of light", [
        "very fast", "10m/s", "the speed of light is C"
    ])
    print(a)  # the speed of light is C

    config = {
        "lang": "en-us",
        "min_conf": 0.4,
        "n_answer": 1
    }
    solver = FlashRankEvidenceSolverPlugin(config)

    text = """Mars is the fourth planet from the Sun. It is a dusty, cold, desert world with a very thin atmosphere. 
Mars is also a dynamic planet with seasons, polar ice caps, canyons, extinct volcanoes, and evidence that it was even more active in the past.
Mars is one of the most explored bodies in our solar system, and it's the only planet where we've sent rovers to roam the alien landscape. 
NASA currently has two rovers (Curiosity and Perseverance), one lander (InSight), and one helicopter (Ingenuity) exploring the surface of Mars.
"""
    query = "how many rovers are currently exploring Mars"
    answer = solver.get_best_passage(evidence=text, question=query)
    print("Query:", query)
    print("Answer:", answer)
    # 2024-07-22 17:08:38.542 - OVOS - ovos_plugin_manager.language:create:233 - INFO - Loaded the Language Translation plugin ovos-translate-plugin-server
    # 2024-07-22 17:08:38.543 - OVOS - ovos_plugin_manager.utils.config:get_plugin_config:40 - DEBUG - Loaded configuration: {'module': 'ovos-translate-plugin-server', 'lang': 'en-us'}
    # 2024-07-22 17:08:38.552 - OVOS - ovos_plugin_manager.language:create:233 - INFO - Loaded the Language Translation plugin ovos-translate-plugin-server
    # 2024-07-22 17:08:38.552 - OVOS - ovos_plugin_manager.utils.config:get_plugin_config:40 - DEBUG - Loaded configuration: {'module': 'ovos-translate-plugin-server', 'lang': 'en-us'}
    # Query: how many rovers are currently exploring Mars
    # Answer: NASA currently has two rovers (Curiosity and Perseverance), one lander (InSight), and one helicopter (Ingenuity) exploring the surface of Mars.

    # Create your corpus here
    corpus = [
        "a cat is a feline and likes to purr",
        "a dog is the human's best friend and loves to play",
        "a bird is a beautiful animal that can fly",
        "a fish is a creature that lives in water and swims",
    ]

    s = FlashRankCorpusSolver({})
    s.load_corpus(corpus)

    query = "does the fish purr like a cat?"
    print(s.spoken_answer(query))

    # 2024-07-22 17:08:38.595 - OVOS - ovos_plugin_manager.language:create:233 - INFO - Loaded the Language Translation plugin ovos-translate-plugin-server
    # 2024-07-22 17:08:38.595 - OVOS - ovos_plugin_manager.utils.config:get_plugin_config:40 - DEBUG - Loaded configuration: {'module': 'ovos-translate-plugin-server', 'lang': 'en-us'}
    # 2024-07-22 17:08:38.605 - OVOS - ovos_plugin_manager.language:create:233 - INFO - Loaded the Language Translation plugin ovos-translate-plugin-server
    # 2024-07-22 17:08:38.605 - OVOS - ovos_plugin_manager.utils.config:get_plugin_config:40 - DEBUG - Loaded configuration: {'module': 'ovos-translate-plugin-server', 'lang': 'en-us'}
    # a cat is a feline and likes to purr