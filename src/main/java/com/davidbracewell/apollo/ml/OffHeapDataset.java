package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.bayes.BernoulliNaiveBayesLearner;
import com.davidbracewell.collection.Collect;
import com.davidbracewell.function.Unchecked;
import com.davidbracewell.io.CSV;
import com.davidbracewell.io.Resources;
import com.davidbracewell.io.resource.Resource;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.Streams;
import com.davidbracewell.string.CSVFormatter;
import com.davidbracewell.string.StringUtils;
import com.google.common.base.Throwables;
import com.google.common.util.concurrent.AtomicDouble;
import lombok.NonNull;

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author David B. Bracewell
 */
public class OffHeapDataset extends BaseDataset {

  private final AtomicLong id = new AtomicLong();
  private final CSVFormatter formatter = CSV.builder().formatter();
  private Resource outputResource = Resources.temporaryDirectory();
  private int size = 0;

  protected OffHeapDataset(FeatureEncoder featureEncoder, LabelEncoder labelEncoder) {
    super(featureEncoder, labelEncoder);
  }

  public static void main(String[] args) {
    Featurizer<String> featurizer = Featurizer.binary(InMemoryDataset::split);

    Dataset dataset = Dataset
      .builder()
      .type(Type.OffHeap)
      .streamSource(
        Streams.range(0, 100).map((id) -> featurizer.extract(StringUtils.randomHexString(7), Math.random() > 0.5 ? "A" : "B"))
      ).build()
      .shuffle();

    dataset.fold(10).forEach(split -> {

      Classifier classifier = new BernoulliNaiveBayesLearner<>(() -> split.getV1().getFeatureEncoder())
        .train(() -> split.getV1().stream());

      AtomicDouble correct = new AtomicDouble(0);
      split.getV2().stream().forEach(instance -> {
        if (classifier.classify(instance).getResult().equals(instance.getLabel())) {
          correct.addAndGet(1);
        }
      });
      System.out.println("Test: " + (correct.doubleValue() / split.getV2().size()));

    });

  }

  @Override
  public void add(Instance instance) {
    try (BufferedWriter writer = new BufferedWriter(outputResource.getChild("part-" + id.incrementAndGet() + ".csv").writer())) {
      writer.write(instanceToString(instance));
      writer.newLine();
      size++;
    } catch (IOException e) {
      throw Throwables.propagate(e);
    }
  }

  @Override
  public void addAll(@NonNull MStream<Instance> stream) {
    try (BufferedWriter writer = new BufferedWriter(outputResource.getChild("part-" + id.incrementAndGet() + ".csv").writer())) {
      for (Instance instance : Collect.asIterable(stream.iterator())) {
        writer.write(instanceToString(instance));
        writer.newLine();
        size++;
      }
    } catch (IOException e) {
      throw Throwables.propagate(e);
    }
  }

  @Override
  public void addAll(Iterable<Instance> instances) {
    try (BufferedWriter writer = new BufferedWriter(outputResource.getChild("part-" + id.incrementAndGet() + ".csv").writer())) {
      for (Instance instance : instances) {
        writer.write(instanceToString(instance));
        writer.newLine();
        size++;
      }
    } catch (IOException e) {
      throw Throwables.propagate(e);
    }
  }

  private Instance stringToInstance(String line) {
    List<String> in = StringUtils.split(line, ',');
    String label = in.get(0);
    List<Feature> features = new LinkedList<>();
    for (int i = 1; i < in.size(); i += 2) {
      features.add(Feature.real(in.get(i), Double.valueOf(in.get(i + 1))));
    }
    return Instance.create(features, label);
  }

  private String instanceToString(Instance instance) {
    List<Object> output = new LinkedList<>();
    output.add(instance.getLabel());
    instance.forEach(f -> {
      output.add(f.getName());
      output.add(f.getValue());
    });
    return formatter.format(output);
  }

  @Override
  public MStream<Instance> stream() {
    return Streams.of(
      outputResource.getChildren().stream()
        .flatMap(Unchecked.function(r -> r.readLines().stream()))
        .map(this::stringToInstance)
    );
  }

  @Override
  public Dataset shuffle() {
    Random random = new Random();
    MStream<Instance> rStream = Streams.of(random.doubles(size()).boxed()).zip(stream())
      .sortByKey(true)
      .map((t, v) -> v);
    return create(rStream, getFeatureEncoder().createNew(), getLabelEncoder().createNew());
  }

  @Override
  public int size() {
    return size;
  }

  @Override
  protected Dataset create(@NonNull MStream<Instance> instances, @NonNull FeatureEncoder featureEncoder, @NonNull LabelEncoder labelEncoder) {
    Dataset dataset = new OffHeapDataset(featureEncoder, labelEncoder);
    dataset.addAll(instances);
    return dataset;
  }

}// END OF OffHeapDataset
