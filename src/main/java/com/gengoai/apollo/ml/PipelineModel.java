package com.gengoai.apollo.ml;

/**
 * @author David B. Bracewell
 */
public abstract class PipelineModel{
//   <T, P extends ModelParameters, LABEL> implements Model<T, P> {
//   private static final long serialVersionUID = 1L;
//   private final Vectorizer<String> featureVectorizer;
//   private final Vectorizer<LABEL> labelVectorizer;
//
//   protected PipelineModel(Vectorizer<String> featureVectorizer,
//                           Vectorizer<LABEL> labelVectorizer
//                          ) {
//      this.featureVectorizer = featureVectorizer;
//      this.labelVectorizer = labelVectorizer;
//   }
//
//
//   public Vectorizer<String> getFeatureVectorizer() {
//      return featureVectorizer;
//   }
//
//   public Vectorizer<LABEL> getLabelVectorizer() {
//      return labelVectorizer;
//   }
//
//   public T transform(Example example) {
//      return transform(featureVectorizer.transform(example));
//   }
//
//   public void fit(Dataset dataset, P parameters) {
//      //TODO
//   }
//
//   public final void fit(Dataset dataset, Consumer<P> parameters) {
//      P actualizedParameters = getDefaultParameters();
//      parameters.accept(actualizedParameters);
//      fit(dataset, actualizedParameters);
//   }
//

}//END OF PipelineModel
