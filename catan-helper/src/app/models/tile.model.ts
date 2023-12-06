export class Tile {
    numberToken: string;
    resource: string;
    location: number;
    port: string;
  
    constructor(numberToken: string, resource: string, location: number, port: string) {
      this.numberToken = numberToken;
      this.resource = resource;
      this.location = location;
      this.port = port;
    }
}
  