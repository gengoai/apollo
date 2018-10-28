package com.gengoai.apollo.linear.store;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.collection.Streams;
import com.gengoai.io.Resources;
import com.gengoai.io.resource.Resource;
import com.gengoai.string.Strings;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.util.stream.Collectors;

/**
 * @author David B. Bracewell
 */
public class VSTextUtils {

   private VSTextUtils() {
      throw new IllegalAccessError();
   }

   public static String vectorToLine(NDArray vec) {
      return Streams.asStream(vec.iterator())
                    .map(v -> Float.toString(v.getValue()))
                    .collect(Collectors.joining(" "));
   }

   public static NDArray convertLineToVector(String line, int dimension) {
      NDArray vector = NDArrayFactory.DENSE.zeros(1, dimension);
      String[] parts = line.split("[ \t]+");
      for (int i = 1; i < parts.length; i++) {
         vector.set(i - 1, Double.parseDouble(parts[i]));
      }
      vector.setLabel(parts[0]);
      return vector;
   }

   public static int determineDimension(File vectorStore) throws IOException {
      Resource r = Resources.fromFile(vectorStore);
      try (BufferedReader reader = new BufferedReader(r.reader())) {
         String line = reader.readLine();
         if (Strings.isNullOrBlank(line)) {
            throw new IOException("Unexpected empty line at beginning of file: " + vectorStore);
         }
         String[] cells = line.trim().split("[ \t]+");
         if (cells.length > 4) {
            return cells.length - 1;
         }
         return Integer.parseInt(cells[1]);
      }
   }

}//END OF VSTextUtils
