package com.davidbracewell.apollo.ml.preprocess;

import com.davidbracewell.apollo.ml.Dataset;
import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.string.StringUtils;

import java.io.Serializable;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * @author David B. Bracewell
 */
public abstract class RestrictedInstancePreprocessor implements Restricted, InstancePreprocessor, Serializable {
  private static final long serialVersionUID = 1L;
  private final String featureNamePrefix;
  private final boolean acceptAll;

  /**
   * Instantiates a new Restricted filter.
   *
   * @param featureNamePrefix the feature name prefix
   */
  protected RestrictedInstancePreprocessor(String featureNamePrefix) {
    this.featureNamePrefix = featureNamePrefix;
    this.acceptAll = StringUtils.isNullOrBlank(featureNamePrefix);
  }

  public final Stream<Feature> filterPositive(Instance example) {
    return example.getFeatures().stream().filter(f -> acceptAll() || f.getName().startsWith(getRestriction()));
  }


  public final Stream<Feature> filterNegative(Instance example) {
    return example.getFeatures().stream().filter(f -> !acceptAll() && !f.getName().startsWith(getRestriction()));
  }

  public boolean requiresFit() {
    return true;
  }

  @Override
  public final void fit(Dataset<Instance> dataset) {
    if (requiresFit()) {
      restrictedFitImpl(dataset.stream().map(i -> filterPositive(i).collect(Collectors.toList())));
    }
  }

  protected abstract void restrictedFitImpl(MStream<List<Feature>> stream);

  @Override
  public boolean acceptAll() {
    return acceptAll;
  }

  @Override
  public String getRestriction() {
    return featureNamePrefix;
  }

  @Override
  public final Instance apply(Instance example) {
    if (acceptAll()) {
      return Instance.create(restrictedProcessImpl(example.stream(), example).collect(Collectors.toList()), example.getLabel());
    }
    return Instance.create(
      Stream.concat(
        restrictedProcessImpl(filterPositive(example), example),
        filterNegative(example)
      ).collect(Collectors.toList()),
      example.getLabel()
    );
  }


  protected abstract Stream<Feature> restrictedProcessImpl(Stream<Feature> featureStream, Instance originalExample);

}// END OF RestrictedInstancePreprocessor
