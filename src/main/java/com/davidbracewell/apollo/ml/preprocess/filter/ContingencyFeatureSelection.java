package com.davidbracewell.apollo.ml.preprocess.filter;

import com.davidbracewell.apollo.ContingencyTable;
import com.davidbracewell.apollo.ContingencyTableCalculator;
import com.davidbracewell.apollo.ml.Dataset;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.preprocess.InstancePreprocessor;
import com.davidbracewell.collection.Counter;
import com.davidbracewell.collection.HashMapCounter;
import com.davidbracewell.collection.HashMapMultiCounter;
import com.davidbracewell.collection.MultiCounter;
import com.davidbracewell.stream.accumulator.MAccumulator;
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
  private final Set<String> finalFeatures = new HashSet<>();
  private final int numFeaturesPerClass;
  private final double threshold;


  public ContingencyFeatureSelection(@NonNull ContingencyTableCalculator calculator, int numFeaturesPerClass, double threshold) {
    this.calculator = calculator;
    this.numFeaturesPerClass = numFeaturesPerClass;
    this.threshold = threshold;
  }


  @Override
  public void fit(@NonNull Dataset<Instance> dataset) {
    MAccumulator<Counter<Object>> labelCounts = dataset.getType().getStreamingContext().counterAccumulator();
    MAccumulator<MultiCounter<String, Object>> featureLabelCounts = dataset.getType().getStreamingContext().multiCounterAccumulator();
    dataset.stream().forEach(instance -> {
      Object label = instance.getLabel();
      labelCounts.add(new HashMapCounter<>(label));
      MultiCounter<String, Object> localCounts = new HashMapMultiCounter<>();
      instance.getFeatureSpace().forEach(f -> localCounts.increment(f, label));
      featureLabelCounts.add(localCounts);
    });

    double totalCount = labelCounts.value().sum();
    for (Object label : labelCounts.value().items()) {
      double labelCount = labelCounts.value().get(label);
      Map<String, Double> featureScores = new HashMap<>();
      featureLabelCounts.value().items().forEach(feature -> {
          double featureLabelCount = featureLabelCounts.value().get(feature, label);
          double featureSum = featureLabelCounts.value().get(feature).sum();
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
    finalFeatures.clear();
  }

  @Override
  public String describe() {
    return getClass().getSimpleName() + ": numberOfFeaturesPerClass=" + numFeaturesPerClass + ", threshold=" + threshold;
  }

  @Override
  public Instance apply(Instance example) {
    example.getFeatures().removeIf(f -> !finalFeatures.contains(f.getName()));
    return example;
  }


}// END OF ContingencyFeatureSelection
