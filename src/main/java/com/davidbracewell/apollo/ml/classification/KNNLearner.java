package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.linalg.CosineDistanceSignature;
import com.davidbracewell.apollo.linalg.InMemoryLSH;
import com.davidbracewell.apollo.linalg.LabeledVector;
import com.davidbracewell.apollo.linalg.SignatureFunction;
import com.davidbracewell.apollo.ml.Dataset;
import com.davidbracewell.apollo.ml.FeatureVector;
import com.davidbracewell.apollo.ml.Instance;
import lombok.NonNull;

import java.util.function.BiFunction;

/**
 * The type Knn learner.
 *
 * @author David B. Bracewell
 */
public class KNNLearner extends ClassifierLearner {
  private static final long serialVersionUID = 1L;
  private KNN model = null;
  private int K = 3;
  private BiFunction<Integer, Integer, SignatureFunction> signatureSupplier = CosineDistanceSignature::new;

  @Override
  protected Classifier trainImpl(Dataset<Instance> dataset) {
    model = new KNN(
      dataset.getEncoderPair(),
      dataset.getPreprocessors()
    );
    model.K = K;
    model.vectors = InMemoryLSH.builder()
      .signatureSupplier(signatureSupplier)
      .dimension(model.getFeatureEncoder().size())
      .createVectorStore();
    dataset.forEach(instance -> {
      FeatureVector fv = instance.toVector(model.getEncoderPair());
      model.vectors.add(new LabeledVector((int) fv.getLabel(), fv));
    });
    return model;
  }

  @Override
  public void reset() {
    model = null;
  }

  /**
   * Gets k.
   *
   * @return the k
   */
  public int getK() {
    return K;
  }

  /**
   * Sets k.
   *
   * @param k the k
   */
  public void setK(int k) {
    K = k;
  }

  /**
   * Gets signature supplier.
   *
   * @return the signature supplier
   */
  public BiFunction<Integer, Integer, SignatureFunction> getSignatureSupplier() {
    return signatureSupplier;
  }

  /**
   * Sets signature supplier.
   *
   * @param signatureSupplier the signature supplier
   */
  public void setSignatureSupplier(@NonNull BiFunction<Integer, Integer, SignatureFunction> signatureSupplier) {
    this.signatureSupplier = signatureSupplier;
  }

}// END OF KNNLearner
