package com.davidbracewell.apollo.ml.preprocess.filter;

import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.preprocess.InstancePreprocessor;
import com.davidbracewell.apollo.stats.ContingencyTable;
import com.davidbracewell.apollo.stats.ContingencyTableCalculator;
import com.davidbracewell.collection.Counter;
import com.davidbracewell.collection.HashMapCounter;
import com.davidbracewell.collection.HashMapMultiCounter;
import com.davidbracewell.collection.MultiCounter;
import lombok.NonNull;

import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * @author David B. Bracewell
 */
public class ContingencyFeatureSelection implements FilterProcessor<Instance>, InstancePreprocessor, Serializable {
  private static final long serialVersionUID = 1L;
  private final ContingencyTableCalculator calculator;
  private final MultiCounter<String, Object> featureLabelCounter = new HashMapMultiCounter<>();
  private final Counter<Object> labelCounter = new HashMapCounter<>();
  private final Set<String> finalFeatures = new HashSet<>();
  private final int numFeaturesPerClass;
  private final double threshold;


  protected ContingencyFeatureSelection(@NonNull ContingencyTableCalculator calculator, int numFeaturesPerClass, double threshold) {
    this.calculator = calculator;
    this.numFeaturesPerClass = numFeaturesPerClass;
    this.threshold = threshold;
  }


  @Override
  public void visit(Instance example) {
    Object label = example.getLabel();
    if (label != null) {
      labelCounter.increment(label);
      example.getFeatures().forEach(f -> featureLabelCounter.increment(f.getName(), label));
    }
  }

  @Override
  public Instance process(Instance example) {
    example.getFeatures().removeIf(f -> !finalFeatures.contains(f.getName()));
    return example;
  }

  @Override
  public void finish() {
    for (Object label : labelCounter.items()) {
      double labelCount = labelCounter.get(label);
      double totalCount = labelCounter.sum();
      Map<String, Double> featureScores = new HashMap<>();
      featureLabelCounter.items().forEach(feature -> {
          double featureLabelCount = featureLabelCounter.get(feature, label);
          double featureSum = featureLabelCounter.get(feature).sum();
          if (featureLabelCount > 0) {
            double score = calculator.calculate(
              ContingencyTable.create2X2(featureLabelCount, labelCount, featureSum, totalCount)
            );
            featureScores.put(feature, score);
          }
        }
      );

      List<Map.Entry<String, Double>> entryList = featureScores.entrySet()
        .stream()
        .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
        .filter(e -> e.getValue() >= threshold)
        .collect(Collectors.toList());

      if (entryList.size() > 0) {
        entryList.subList(0, Math.min(numFeaturesPerClass, entryList.size())).forEach(e -> finalFeatures.add(e.getKey()));
      }
    }
  }

  @Override
  public void reset() {
    featureLabelCounter.clear();
    labelCounter.clear();
    finalFeatures.clear();
  }

  @Override
  public String describe() {
    return getClass().getSimpleName() + ": numberOfFeaturesPerClass=" + numFeaturesPerClass + ", threshold=" + threshold;
  }

}// END OF ContingencyFeatureSelection
