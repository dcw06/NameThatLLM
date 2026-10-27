"""Topic clusters are label-free and hold out whole topics in every outer and inner split."""

import json

import numpy as np
import pandas as pd
import pytest

from ntl import ensemble, topic_cv
from ntl.modeling import BaselineConfig
from ntl.smoke import synthetic_frames

TOPICS = [["galaxy", "stars", "cosmos", "orbit", "nebula"], ["cells", "protein", "genome", "tissue", "enzyme"],
          ["poetry", "verse", "rhyme", "meter", "stanza"]]


@pytest.fixture
def topical():
    train, _ = synthetic_frames(n_prompts=40)
    prompt_number = train["prompt_id"].str[-3:].astype(int)
    rng = np.random.default_rng(0)
    train["text"] = [f"{text} " + " ".join(rng.choice(TOPICS[number % 3], size=4))
                     for text, number in zip(train["text"], prompt_number, strict=True)]
    return train, prompt_number % 3


def test_topics_ignore_labels_and_recover_planted_topics(topical):
    train, planted = topical
    topics = topic_cv.prompt_topics(train)
    shuffled = topic_cv.prompt_topics(train.assign(model=train["model"].sample(frac=1, random_state=1).to_numpy()))
    np.testing.assert_array_equal(topics["similarity"], shuffled["similarity"])
    groups = topic_cv.topic_clusters(topics, 3)
    truth = dict(zip(train["prompt_id"], planted, strict=True))
    clusters = pd.DataFrame({"truth": [truth[p] for p in groups], "cluster": list(groups.values())})
    assert clusters.groupby("cluster")["truth"].nunique().max() == 1
    described = topic_cv.describe_clusters(topics, groups)
    assert sum(item["prompts"] for item in described.values()) == 40


def test_run_holds_out_whole_topics_and_reports_every_condition(topical, tmp_path):
    train, _ = topical
    features = np.random.default_rng(1).normal(size=(len(train), 6)).astype(np.float32)
    config = ensemble.BlendConfig(cpu=BaselineConfig(char_max_features=300, word_max_features=100, max_iter=200),
                                  bootstrap_samples=20)
    summary = topic_cv.run(train, features, tmp_path / "topic", ks=(6,), config=config)
    assert set(summary["conditions"]) == {"rows", "prompts", "topics-6"}
    assert summary["conditions"]["rows"]["median_nearest_training_prompt_similarity"] == 1.0
    folds = pd.read_csv(tmp_path / "topic" / "topics-6" / "folds" / "folds.csv")
    assert folds.groupby("prompt_id")["fold"].nunique().max() == 1  # prompt_id here is the topic group
    assert summary["conditions"]["topics-6"]["groups"] == 6
    assert "terciles" in summary["conditions"]["prompts"]["by_topic_distance"]
    # JSON stores the configuration tuples as lists.
    assert json.loads((tmp_path / "topic" / "summary.json").read_text()) == json.loads(json.dumps(summary))
