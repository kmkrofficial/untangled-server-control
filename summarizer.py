from transformers import pipeline
import os


class Summarizer:
    def __init__(self):
        self.instant_history = None
        self.summarizer_model = pipeline("summarization")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    def precis_writing(self, txt, max_length=100, min_length=5):
        summary_text = self.summarizer_model(txt, max_length=max_length, min_length=min_length, do_sample=False)[0]["summary_text"]
        self.instant_history = summary_text
        return summary_text

    def return_instant_history(self):
        return self.instant_history


text = """One month after the United States began what has become a troubled rollout of a national COVID vaccination 
campaign, the effort is finally gathering real steam. Close to a million doses -- over 951,000, to be more exact -- 
made their way into the arms of Americans in the past 24 hours, the U.S. Centers for Disease Control and Prevention 
reported Wednesday. That's the largest number of shots given in one day since the rollout began and a big jump from 
the previous day, when just under 340,000 doses were given, CBS News reported. That number is likely to jump quickly 
after the federal government on Tuesday gave states the OK to vaccinate anyone over 65 and said it would release all 
the doses of vaccine it has available for distribution. Meanwhile, a number of states have now opened mass 
vaccination sites in an effort to get larger numbers of people inoculated, CBS News reported. """
