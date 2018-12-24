package com.gengoai.apollo.ml.vectorizer;

import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.collection.HashMapIndex;
import com.gengoai.collection.Index;
import com.gengoai.collection.Sets;
import com.gengoai.collection.Streams;
import com.gengoai.json.JsonEntry;

import java.util.Set;

/**
 * @author David B. Bracewell
 */
public class IndexVectorizer extends StringVectorizer {
   private static final long serialVersionUID = 1L;
   private final Index<String> index = new HashMapIndex<>();
   private final boolean isLabel;
   private final String unknown;

   public IndexVectorizer(boolean isLabel) {
      this(isLabel, null);
   }


   public IndexVectorizer(boolean isLabel, String unknown) {
      super(isLabel);
      this.isLabel = isLabel;
      this.unknown = unknown;
      if (this.unknown != null) {
         index.add(this.unknown);
      }
   }

   public static IndexVectorizer featureVectorizer() {
      return new IndexVectorizer(false);
   }

   public static IndexVectorizer featureVectorizer(String unknownFeature) {
      return new IndexVectorizer(false, unknownFeature);
   }

   public static IndexVectorizer labelVectorizer() {
      return new IndexVectorizer(true);
   }

   @Override
   public Set<String> alphabet() {
      return Sets.asHashSet(index);
   }

   @Override
   public String decode(double value) {
      return index.get((int) value);
   }

   @Override
   public double encode(String value) {
      int i = index.getId(value);
      if (i < 0 && unknown != null) {
         return index.getId(unknown);
      }
      return i;
   }

   @Override
   public void fit(Dataset dataset) {
      index.clear();
      index.addAll(dataset.stream()
                          .flatMap(Streams::asStream)
                          .flatMap(example -> {
                             if (isLabel) {
                                return example.getStringLabelSpace();
                             }
                             return example.getFeatureNameSpace();
                          })
                          .distinct()
                          .collect());
   }

   @Override
   public int size() {
      return index.size();
   }

   public JsonEntry toJson() {
      return JsonEntry.object()
                      .addProperty("class", IndexVectorizer.class)
                      .addProperty("isLabel", isLabel)
                      .addProperty("index", index);
   }

   @Override
   public String toString() {
      return "IndexVectorizer{" +
                "size=" + index.size() +
                ", vectorizing=" + (isLabel ? "Label" : "Feature") +
                '}';
   }
}//END OF IndexEncoder
