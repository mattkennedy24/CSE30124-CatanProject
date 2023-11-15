export class Tile {
    numberToken: number;
    resource: string;
    location: number;
  
    constructor(numberToken: number, resource: string, location: number) {
      this.numberToken = numberToken;
      this.resource = resource;
      this.location = location;
    }
}
  