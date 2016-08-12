package com.davidbracewell.apollo.ml.preprocess;

import com.clearspring.analytics.util.Preconditions;
import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.string.StringUtils;
import lombok.NonNull;

import java.io.Serializable;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * The type Restricted instance preprocessor.
 *
 * @author David B. Bracewell
 */
public abstract class RestrictedInstancePreprocessor implements Restricted, InstancePreprocessor, Serializable {
  private static final long serialVersionUID = 1L;
  private String featureNamePrefix;
  private boolean acceptAll;

  /**
   * Instantiates a new Restricted filter.
   *
   * @param featureNamePrefix the feature name prefix
   */
  protected RestrictedInstancePreprocessor(String featureNamePrefix) {
    this.featureNamePrefix = featureNamePrefix;
    this.acceptAll = StringUtils.isNullOrBlank(featureNamePrefix);
  }

  /**
   * Instantiates a new Restricted instance preprocessor.
   */
  protected RestrictedInstancePreprocessor() {
    this.featureNamePrefix = null;
    this.acceptAll = true;
  }

  /**
   * Filter positive stream.
   *
   * @param example the example
   * @return the stream
   */
  public final Stream<Feature> filterPositive(Instance example) {
    return example.getFeatures().stream().filter(f -> acceptAll() || f.getName().startsWith(getRestriction()));
  }


  /**
   * Filter negative stream.
   *
   * @param example the example
   * @return the stream
   */
  public final Stream<Feature> filterNegative(Instance example) {
    return example.getFeatures().stream().filter(f -> !acceptAll() && !f.getName().startsWith(getRestriction()));
  }

  @Override
  public final void fit(Dataset<Instance> dataset) {
    if (requiresFit()) {
      restrictedFitImpl(dataset.stream().map(i -> filterPositive(i).collect(Collectors.toList())));
    }
  }

  /**
   * Restricted fit.
   *
   * @param stream the stream
   */
  protected abstract void restrictedFitImpl(MStream<List<Feature>> stream);

  @Override
  public boolean acceptAll() {
    return acceptAll;
  }

  @Override
  public String getRestriction() {
    return featureNamePrefix;
  }

  /**
   * Sets restriction.
   *
   * @param prefix the prefix
   */
  protected final void setRestriction(String prefix) {
    this.featureNamePrefix = prefix;
    this.acceptAll = StringUtils.isNullOrBlank(featureNamePrefix);
  }

  @Override
  public final Instance apply(Instance example) {
    if (acceptAll()) {
      return Instance.create(restrictedProcessImpl(example.stream(), example).collect(Collectors.toList()),
                             example.getLabel());
    }
    return Instance.create(
      Stream.concat(
        restrictedProcessImpl(filterPositive(example), example),
        filterNegative(example)
      ).collect(Collectors.toList()),
      example.getLabel()
    );
  }


  /**
   * Restricted process stream.
   *
   * @param featureStream   the feature stream
   * @param originalExample the original example
   * @return the stream
   */
  protected abstract Stream<Feature> restrictedProcessImpl(Stream<Feature> featureStream, Instance originalExample);

  @Override
  public String toString() {
    return describe();
  }

}// END OF RestrictedInstancePreprocessor
