package com.davidbracewell.apollo.ml.preprocess.transform;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.preprocess.InstancePreprocessor;
import com.davidbracewell.apollo.ml.preprocess.TransformProcessor;
import com.davidbracewell.string.StringUtils;
import lombok.NonNull;

import java.io.Serializable;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * @author David B. Bracewell
 */
public abstract class RestrictedTransform implements TransformProcessor<Instance>, InstancePreprocessor, Serializable {
  private static final long serialVersionUID = 1L;
  private final String featureNamePrefix;
  private final boolean acceptAll;

  protected RestrictedTransform(String featureNamePrefix) {
    this.featureNamePrefix = featureNamePrefix;
    this.acceptAll = StringUtils.isNullOrBlank(featureNamePrefix);
  }

  private Stream<Feature> filterPositive(Instance example) {
    return example.getFeatures().stream().filter(f -> acceptAll || f.getName().startsWith(featureNamePrefix));
  }

  private Stream<Feature> filterNegative(Instance example) {
    return example.getFeatures().stream().filter(f -> !acceptAll && !f.getName().startsWith(featureNamePrefix));
  }

  @Override
  public final void visit(Instance example) {
    if (example != null) {
      visitImpl(filterPositive(example));
    }
  }

  protected abstract void visitImpl(Stream<Feature> featureStream);


  @Override
  public Instance process(@NonNull Instance example) {
    if (acceptAll) {
      return Instance.create(
        processImpl(filterPositive(example)).collect(Collectors.toList()),
        example.getLabel()
      );
    }
    return Instance.create(
      Stream.concat(
        processImpl(filterPositive(example)),
        filterNegative(example)
      ).collect(Collectors.toList()),
      example.getLabel()
    );
  }

  protected abstract Stream<Feature> processImpl(Stream<Feature> featureStream);

}// END OF RestrictedTransform
