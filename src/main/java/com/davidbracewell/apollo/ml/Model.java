package com.davidbracewell.apollo.ml;

import com.davidbracewell.io.resource.Resource;
import lombok.NonNull;

import java.io.Serializable;

/**
 * The type Model.
 *
 * @author David B. Bracewell
 */
public abstract class Model implements Serializable {
  private static final long serialVersionUID = 1L;
  private final Encoder labelEncoder;
  private final Encoder featureEncoder;

  /**
   * Instantiates a new Model.
   *
   * @param labelEncoder   the label encoder
   * @param featureEncoder the feature encoder
   */
  public Model(@NonNull Encoder labelEncoder, @NonNull Encoder featureEncoder) {
    this.labelEncoder = labelEncoder;
    this.featureEncoder = featureEncoder;
  }

  /**
   * Read model classifier.
   *
   * @param modelResource the model resource
   * @return the classifier
   * @throws Exception the exception
   */
  public static <T extends Model> T readModel(@NonNull Resource modelResource) throws Exception {
    return modelResource.readObject();
  }

  /**
   * Number of labels int.
   *
   * @return the int
   */
  public int numberOfLabels() {
    return labelEncoder.size();
  }

  /**
   * Number of features int.
   *
   * @return the int
   */
  public int numberOfFeatures() {
    return featureEncoder.size();
  }

  /**
   * Gets label encoder.
   *
   * @return the label encoder
   */
  public Encoder labelEncoder() {
    return labelEncoder;
  }

  /**
   * Gets feature encoder.
   *
   * @return the feature encoder
   */
  public Encoder featureEncoder() {
    return featureEncoder;
  }

  /**
   * Write model.
   *
   * @param modelResource the model resource
   * @throws Exception the exception
   */
  public void writeModel(@NonNull Resource modelResource) throws Exception {
    modelResource.writeObject(this);
  }


}// END OF Model
