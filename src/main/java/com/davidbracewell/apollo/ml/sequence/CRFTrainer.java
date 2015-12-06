package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.Dataset;
import com.davidbracewell.apollo.ml.Instance;
import com.github.jcrfsuite.util.CrfSuiteLoader;
import third_party.org.chokkan.crfsuite.*;

/**
 * @author David B. Bracewell
 */
public class CRFTrainer extends SequenceLabelerLearner {

  @Override
  protected SequenceLabeler trainImpl(Dataset<Sequence> dataset) {
    try {
      CrfSuiteLoader.load();
    } catch (Exception e) {
      e.printStackTrace();
    }

    Trainer trainer = new Trainer();

    dataset.forEach(sequence -> {
      ItemSequence seq = new ItemSequence();
      StringList lables = new StringList();
      for (Instance instance : sequence.asInstances()) {
        Item item = new Item();
        instance.forEach(f -> item.add(new Attribute(f.getName(), f.getValue())));
        lables.add(instance.getLabel().toString());
        seq.add(item);
      }
      trainer.append(seq, lables, 0);
    });

    trainer.select("lbfgs", "crf1d");
    trainer.set("max_iterations", "50");


    trainer.train("/home/david/model.crf", -1);
    return new CRFTagger(
      dataset.getLabelEncoder(),
      dataset.getFeatureEncoder(),
      dataset.getPreprocessors(),
      TransitionFeatures.FIRST_ORDER,
      "/home/david/model.crf"
    );
  }

  @Override
  public void reset() {

  }

}// END OF CRFTrainer
