package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.ClassifierResult;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.github.jcrfsuite.CrfTagger;
import com.github.jcrfsuite.util.CrfSuiteLoader;
import com.github.jcrfsuite.util.Pair;
import lombok.NonNull;
import third_party.org.chokkan.crfsuite.Attribute;
import third_party.org.chokkan.crfsuite.Item;
import third_party.org.chokkan.crfsuite.ItemSequence;

import java.util.List;

/**
 * @author David B. Bracewell
 */
public class CRFTagger extends SequenceLabeler {
  private final String modelFile;
  private volatile CrfTagger tagger;

  /**
   * Instantiates a new Model.
   *
   * @param labelEncoder       the label encoder
   * @param featureEncoder     the feature encoder
   * @param preprocessors      the preprocessors
   * @param transitionFeatures the transition features
   * @param modelFile
   */
  public CRFTagger(@NonNull Encoder labelEncoder, @NonNull Encoder featureEncoder, @NonNull PreprocessorList<Sequence> preprocessors, TransitionFeatures transitionFeatures, String modelFile) {
    super(labelEncoder, featureEncoder, preprocessors, transitionFeatures);
    this.modelFile = modelFile;
    this.tagger = new CrfTagger(modelFile);
  }

  @Override
  public LabelingResult label(@NonNull Sequence sequence) {
    try {
      if (!CrfSuiteLoader.isNativeLibraryLoaded()) {
        CrfSuiteLoader.load();
      }
    } catch (Exception e) {
      e.printStackTrace();
    }
    ItemSequence seq = new ItemSequence();
    for (Instance instance : sequence.asInstances()) {
      Item item = new Item();
      instance.forEach(f -> item.add(new Attribute(f.getName(), f.getValue())));
      seq.add(item);
    }
    List<Pair<String, Double>> tags = tagger.tag(seq);
    LabelingResult lr = new LabelingResult(sequence.size());
    for (int i = 0; i < tags.size(); i++) {
      lr.setLabel(i, tags.get(i).first, tags.get(i).second);
    }
    return lr;
  }

  @Override
  public ClassifierResult estimateInstance(Instance instance) {
    return null;
  }

}// END OF CRFTagger
