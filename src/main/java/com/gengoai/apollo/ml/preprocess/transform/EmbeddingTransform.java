package com.gengoai.apollo.ml.preprocess.transform;

import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.preprocess.RestrictedInstancePreprocessor;
import com.gengoai.stream.MStream;

import java.util.List;
import java.util.stream.Stream;

/**
 * @author David B. Bracewell
 */
public class EmbeddingTransform extends RestrictedInstancePreprocessor implements TransformProcessor<Instance> {
   @Override
   public String describe() {
      return null;
   }

   @Override
   public void reset() {

   }

   @Override
   protected void restrictedFitImpl(MStream<List<Feature>> stream) {

   }

   @Override
   protected Stream<Feature> restrictedProcessImpl(Stream<Feature> featureStream, Instance originalExample) {
      return null;
   }
//   private static final long serialVersionUID = 1L;
//   private final SerializableSupplier<Embedding> embeddingSupplier;
//   private final VectorComposition composition;
//   private final String embeddingFeaturePrefix;
//   private volatile transient Embedding embedding;
//
//   public EmbeddingTransform(String featureNamePrefix, @NonNull SerializableSupplier<Embedding> embeddingSupplier, @NonNull VectorComposition composition, String embeddingFeaturePrefix) {
//      super(featureNamePrefix);
//      this.embeddingSupplier = embeddingSupplier;
//      this.composition = composition;
//      this.embeddingFeaturePrefix = embeddingFeaturePrefix;
//   }
//
//   public EmbeddingTransform(@NonNull SerializableSupplier<Embedding> embeddingSupplier, @NonNull VectorComposition composition, String embeddingFeaturePrefix) {
//      this.embeddingSupplier = embeddingSupplier;
//      this.composition = composition;
//      this.embeddingFeaturePrefix = embeddingFeaturePrefix;
//   }
//
//   @Override
//   public String describe() {
//      return "EmbeddingTransform";
//   }
//
//   Embedding getEmbedding() {
//      if (embedding == null) {
//         synchronized (this) {
//            if (embedding == null) {
//               embedding = embeddingSupplier.get();
//            }
//         }
//      }
//      return embedding;
//   }
//
//   @Override
//   public boolean requiresFit() {
//      return false;
//   }
//
//   @Override
//   public void reset() {
//
//   }
//
//   @Override
//   protected void restrictedFitImpl(MStream<List<Feature>> stream) {
//
//   }
//
//   @Override
//   protected Stream<Feature> restrictedProcessImpl(Stream<Feature> featureStream, Instance originalExample) {
//      Embedding embedding = getEmbedding();
//      return asStream(composition.compose(embedding.dimension(),
//                                          featureStream.map(Feature::getName)
//                                                       .filter(embedding::contains)
//                                                       .map(embedding::get)
//                                                       .collect(Collectors.toList()))
//                                 .nonZeroIterator())
//                .map(e -> Feature.real(embeddingFeaturePrefix + "-" + e.getIndex(), e.getValue()));
//   }


}// END OF EmbeddingTransform
