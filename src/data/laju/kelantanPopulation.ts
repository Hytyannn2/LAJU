export interface JajahanPopulation {
  name: string;
  population: number;
}

export const KELANTAN_POPULATION: Record<string, number> = {
  "Kota Bharu": 590100,
  "Pasir Mas": 242700,
  "Tumpat": 191200,
  "Bachok": 167600,
  "Tanah Merah": 159900,
  "Pasir Puteh": 144700,
  "Machang": 117700,
  "Kuala Krai": 112400,
  "Gua Musang": 111300,
  "Jeli": 58200,
  "Lojing": 11700,
};

export const TOTAL_KELANTAN_POPULATION = Object.values(KELANTAN_POPULATION).reduce(
  (sum, pop) => sum + pop,
  0
);
